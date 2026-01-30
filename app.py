import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
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

st.set_page_config(page_title="KOSPI 정밀 진단 v2.6", layout="wide")

# [데이터 수집] 개별 수집으로 멀티인덱스 에러 방지
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
            raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
            if not raw.empty:
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

# [사용자 정의 날짜 포맷터]
def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    if dt.month == 1:
        return dt.strftime('%Y/%m')
    else:
        return dt.strftime('%m')

try:
    df = load_expert_data()
    model, contribution_pct = get_analysis(df)
    
    # 상단 정보 섹션
    c1, c2 = st.columns([1, 1.5])
    with c1:
        current_chg = (df.iloc[-1] / df.iloc[-2] - 1)
        pred_input = [1] + [current_chg[f] for f in contribution_pct.index]
        pred_val = model.predict(pred_input)[0]
        color = "#e74c3c" if pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #555;">종합 투자 예측 지수</h3>
                <h1 style="color: {color}; font-size: 45px; margin: 10px 0;">{pred_val:+.2%}</h1>
                <p style="color: #666; font-size: 14px;">본 수치는 8대 지표를 기반으로 한 <b>KOSPI 기대 수익률</b>입니다.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("📊 지표별 KOSPI 영향력 비중")
        st.table(pd.DataFrame(contribution_pct).T.style.format("{:.1f}%"))

    st.divider()

    # 하단 그래프 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    plt.subplots_adjust(hspace=0.6)

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
        plot_data = df[col].tail(100) # 더 넓은 시계열로 날짜 변화 확인
        
        # 위험선 계산
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std

        # 시각화
        ax.plot(plot_data, color='#34495e', lw=2.5)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        
        # [지능형 날짜 포맷 적용]
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator()) # 월 단위로 눈금 표시
        
        # 위험선 근거 표기
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=10)
        ax.text(plot_data.index[0], threshold, f"근거: {th_label}", 
                fontproperties=fprop, color='#e74c3c', va='bottom', fontsize=10, backgroundcolor='#ffffff')

        # 전문 진단 설명 (하단 xlabel 활용)
        safe_th = threshold if threshold != 0 else 1
        dist = abs(plot_data.iloc[-1] - threshold) / abs(safe_th)
        ax.set_xlabel(f"위험선 대비 거리: {dist:.1%} | 이탈 시 [{warn_text}]", fontproperties=fprop, fontsize=11, color='#555')
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    st.pyplot(fig)

except Exception as e:
    st.error(f"오류 발생: {e}")
