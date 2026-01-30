import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분 주기
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 정밀 진단 시스템", layout="wide")

# [데이터 수집] 개별 수집을 통해 멀티인덱스 에러 방지
@st.cache_data(ttl=300)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y'
    }
    
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    combined_df = pd.DataFrame()

    for ticker, name in tickers.items():
        try:
            # 과거 데이터와 실시간 데이터를 안전하게 개별 수집
            raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
            if not raw.empty:
                # 최신 장중가 업데이트
                rt = yf.download(ticker, period='1d', interval='1m', progress=False)
                val = rt['Close'].iloc[-1] if not rt.empty else raw['Close'].iloc[-1]
                
                series = raw['Close'].copy()
                series.iloc[-1] = val
                combined_df[name] = series
        except:
            continue

    df = combined_df.ffill().interpolate()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    
    return df.dropna().tail(300)

# [분석] 영향도 100% 산출
def get_analysis(df):
    returns = np.log(df / df.shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    y = returns['KOSPI']
    X = (returns[features] - returns[features].mean()) / returns[features].std()
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    abs_coeffs = np.abs(model.params.drop('const'))
    contribution = (abs_coeffs / abs_coeffs.sum()) * 100
    return model, contribution

# [UI 구현]
st.title("🏛️ KOSPI 8대 지표 정밀 진단 시스템")

try:
    df = load_expert_data()
    model, contribution_pct = get_analysis(df)
    
    # 상단 예측 및 비중 표
    c1, c2 = st.columns([1, 1.5])
    with c1:
        current_chg = (df.iloc[-1] / df.iloc[-2] - 1)
        pred_input = [1] + [current_chg[f] for f in contribution_pct.index]
        pred_val = model.predict(pred_input)[0]
        
        st.metric("종합 투자 예측 지수 (기대수익률)", f"{pred_val:+.2%}")
        st.write("**💡 수치 해석:** 8대 지표의 에너지를 종합한 코스피 방향성입니다.")
        
    with c2:
        st.subheader("📊 지표별 KOSPI 영향력 비중")
        st.table(pd.DataFrame(contribution_pct).T.style.format("{:.1f}%"))

    st.divider()

    # 하단 8대 지표 그래프 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    config = [
        ('KOSPI', '1. KOSPI 본체', 'MA250 - 1σ', '장기 추세 붕괴'),
        ('Exchange', '2. 원/달러 환율', 'MA250 + 1.5σ', '외인 자금 이탈'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', 'IT 공급망 위기'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '글로벌 심리 위축'),
        ('VIX', '5. 공포지수(VIX)', '20.0', '시장 패닉 진입'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '아시아권 경기 침체'),
        ('Yield_Spread', '7. 장단기 금리차', '0.0', '경제 불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '유동성 긴축 압박')
    ]

    for i, (col, title, th_label, warn_text) in enumerate(config):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(60)
        
        # 위험선 계산
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std

        # 시각화
        ax.plot(plot_data, color='#34495e', lw=2)
        ax.axhline(y=threshold, color='#e74c3c', ls='--')
        
        # 위험선 근거 표기
        ax.set_title(title, fontproperties=fprop, fontsize=14, fontweight='bold')
        ax.text(plot_data.index[0], threshold, f"근거: {th_label}", 
                fontproperties=fprop, color='#e74c3c', va='bottom', fontsize=10)

        # 전문 진단 텍스트 (단순화 버전)
        dist = abs(plot_data.iloc[-1] - threshold) / (abs(threshold) if threshold != 0 else 1)
        ax.set_xlabel(f"위험선까지 거리: {dist:.1%}\n이탈 시 [{warn_text}] 판단", fontproperties=fprop, fontsize=10)
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"오류 발생: {e}")
