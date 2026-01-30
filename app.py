import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

# [설정] 페이지 레이아웃 가로 확장
st.set_page_config(page_title="KOSPI 8대 요인 복합 진단", layout="wide")

# [데이터 수집]
@st.cache_data(ttl=3600)
def load_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
    data = data.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    return data.dropna()

# [분석] 회귀 모델링
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("🛡️ KOSPI 8대 핵심 요인 복합 진단 시스템")
st.markdown("한 줄에 4개씩, 총 8개 지표를 가로로 배치하여 최근 데이터 기반 위험선을 모니터링합니다.")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # 상단 요약 지표
    col_a, col_b, col_c = st.columns(3)
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    with col_a: st.metric("모델 설명력 (R²)", f"{model.rsquared:.2%}")
    with col_b: 
        status = "위험" if pred < -0.003 else "경계" if pred < 0 else "안정"
        st.subheader(f"종합 진단: {status}")
    with col_c: st.write(f"최근 데이터: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # [그래프 섹션] 2행 4열 구조
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    plt.rcParams['axes.unicode_minus'] = False

    # 지표 리스트 및 위험 설정 (최근 데이터 기반)
    # 각 요소: (데이터컬럼, 제목, 위험선, 색상, 설명)
    plot_info = [
        ('KOSPI', '1. KOSPI 지수', 2400, 'black', '심리적 지지선: 2,400'),
        ('Exchange', '2. 환율 (USD/KRW)', 1380, 'tab:blue', '위험 임계점: 1,380원'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 4500, 'tab:green', '공급망 우려선: 4,500'),
        ('SP500', '4. 미 S&P 500', 5500, 'tab:cyan', '추세 이탈선: 5,500'),
        ('VIX', '5. 공포지수(VIX)', 20, 'tab:purple', '공포 확산선: 20.0'),
        ('China', '6. 상하이 종합', 2900, 'tab:red', '경기 침체선: 2,900'),
        ('Yield_Spread', '7. 장단기 금리차', 0, 'tab:orange', '경기 불황선: 0.00'),
        ('US10Y', '8. 미 국채 10Y', 4.5, 'tab:brown', '고금리 압박선: 4.5%')
    ]

    for i, (col, title, threshold, color, desc) in enumerate(plot_info):
        ax = axes[i // 4, i % 4]
        ax.plot(df[col].tail(100), color=color, lw=2)
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold')
        
        # 그래프별 하단 설명 추가
        ax.annotate(desc, xy=(0.5, -0.15), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='red')
        
        # 눈금 폰트 설정
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    
    # 하단 8대 지표 상세 설명 가이드
    st.subheader("📝 지표별 최근 위험 기준 근거")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.write("**1. KOSPI:** 최근 하락 추세에서 강력한 심리적/기술적 지지선인 2,400선을 기준으로 잡았습니다.")
        st.write("**2. 환율:** '뉴노멀' 환율 환경을 반영하여 외국인 수급이 발작하는 1,380원을 기준선으로 설정했습니다.")
    with g2:
        st.write("**3. 미 반도체:** 글로벌 AI 업황의 둔화 여부를 판가름하는 SOX 지수 4,500선을 경계선으로 봅니다.")
        st.write("**4. S&P 500:** 미 증시의 중장기 상승 추세 유지 여부를 결정짓는 5,500선을 기준으로 합니다.")
    with g3:
        st.write("**5. VIX:** 시장 변동성이 평시를 벗어나 패닉으로 진입하는 통계적 수치 20.0을 위험선으로 설정했습니다.")
        st.write("**6. 상하이:** 대중국 수출 의존도를 고려, 중국 경기의 마지노선인 상하이 2,900선을 주시합니다.")
    with g4:
        st.write("**7. 금리차:** 수익률 곡선 역전 후 해소되는 과정에서의 경기 불황 전조인 0.00선을 기준으로 합니다.")
        st.write("**8. 미 10년물:** 고금리 기조가 국내 증시의 밸류에이션을 압박하기 시작하는 4.5%를 경계선으로 잡았습니다.")

except Exception as e:
    st.error(f"데이터 분석 중 오류가 발생했습니다: {e}")
