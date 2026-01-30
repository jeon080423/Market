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

# [설정] 페이지 레이아웃 와이드 모드
st.set_page_config(page_title="KOSPI 8대 지표 표준화 예측 대시보드", layout="wide")

# [데이터 수집]
@st.cache_data(ttl=3600)
def load_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000) # 표준화를 위해 충분한 과거 데이터 수집
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
    data = data.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    return data.dropna()

# [분석] 회귀 모델링 (8대 지표 복합 분석)
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("📊 KOSPI 8대 지표 표준화 예측 대시보드")
st.markdown("절대값이 아닌 **최근 1년 변동성($\sigma$) 및 이동평균**을 기준으로 인플레이션이 반영된 상대적 위험도를 측정합니다.")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # 상단 요약 지표
    col_a, col_b, col_c = st.columns(3)
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    with col_a: st.metric("모델 설명력 (R²)", f"{model.rsquared:.2%}")
    with col_b: 
        status = "하락 경계" if pred < -0.003 else "중립" if pred < 0.001 else "상승 기대"
        st.subheader(f"종합 예측 신호: {status}")
    with col_c: st.write(f"최근 데이터: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # [그래프 섹션] 2행 4열
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    # 표준화된 위험 지표 설정 정보
    # (컬럼명, 제목, 위험조건, 설명)
    # 위험조건: 최근 250일(1년) 이동평균 대비 표준편차 배수 등으로 자동 산출
    plot_items = [
        ('KOSPI', '1. KOSPI 지수', 'MA250 - 1σ', '최근 1년 평균 하단 이탈'),
        ('Exchange', '2. 환율 (USD/KRW)', 'MA250 + 1.5σ', '최근 1년 평균 대비 상단 돌파'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', 'AI 업황 단기 저점 경계'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '미 증시 추세 훼손 우려'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (고정)', '시장 심리 패닉 구간'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '중국 경기 침체 가속'),
        ('Yield_Spread', '7. 장단기 금리차', '0.00 (고정)', '경기 불황 진입 신호'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '고금리 밸류에이션 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(plot_items):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(120) # 최근 약 6개월간의 흐름 시각화
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        # 동적 위험선 계산 (절대값이 아닌 통계적 수치)
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: 
            threshold = 20.0 if col == 'VIX' else 0.0 # 특정 지표는 절대 기준 유지
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='navy', lw=2)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.8, lw=2)
        
        # 1. 그래프 위에 위험선 설명 텍스트 표시
        ax.text(plot_data.index[5], threshold, f" 위험 기준: {threshold_label}", 
                fontproperties=fprop, fontsize=11, color='crimson', 
                verticalalignment='bottom', backgroundcolor='#ffecec')

        # 2. 제목 및 눈금 설정
        ax.set_title(title, fontproperties=fprop, fontsize=15, fontweight='bold', pad=15)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)
        
        # 하단 텍스트 설명
        ax.annotate(f"[{desc}]", xy=(0.5, -0.18), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='#333333')

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**💡 표준화 분석 가이드:** 본 대시보드는 각 지표의 1년 이동평균($\mu$)과 표준편차($\sigma$)를 활용합니다. 붉은 점선은 단순 가격이 아니라 최근 1년 시장이 받아들인 변동 범위를 벗어나는 통계적 '이상치' 구간을 의미합니다.")

except Exception as e:
    st.error(f"데이터 분석 중 오류가 발생했습니다: {e}")
