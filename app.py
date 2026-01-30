import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates  # 날짜 포맷 최적화를 위해 추가
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 정밀 진단 시스템 v2.2", layout="wide")

# [데이터 수집 및 보정] 
@st.cache_data(ttl=300)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    hist_raw = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)
    hist_data = hist_raw['Close'] if isinstance(hist_raw.columns, pd.MultiIndex) else hist_raw
    
    current_prices = {}
    for t in tickers.keys():
        try:
            rt_data = yf.download(t, period='1d', interval='1m', progress=False)
            if not rt_data.empty:
                val = rt_data['Close'].iloc[-1]
                prev_val = hist_data[t].dropna().iloc[-1]
                current_prices[t] = val if abs((val - prev_val) / prev_val) < 0.1 else prev_val
            else:
                current_prices[t] = hist_data[t].dropna().iloc[-1]
        except:
            current_prices[t] = hist_data[t].dropna().iloc[-1]

    df = hist_data.copy()
    today_ts = pd.Timestamp(datetime.now().date())
    if df.index[-1].date() == today_ts.date():
        for t, price in current_prices.items(): df.at[df.index[-1], t] = price
    else:
        new_row = pd.DataFrame([current_prices], index=[pd.Timestamp(datetime.now())])
        df = pd.concat([df, new_row])

    df = df.rename(columns=tickers).ffill().interpolate(method='linear')
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = (df['US10Y'] - df['US2Y'])
    return df.dropna().tail(300)

# [분석] 
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
st.title("🏛️ KOSPI 8대 지표 정밀 진단 시스템 v2.2")

try:
    df = load_expert_data()
    model, contribution_pct = get_analysis(df)
    
    # 상단 요약 섹션
    c1, c2 = st.columns([1, 1.5])
    with c1:
        current_chg = (df.iloc[-1] / df.iloc[-2] - 1)
        pred_val = model.predict([1] + [current_chg[f] for f in contribution_pct.index])[0]
        color = "#e74c3c" if pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 25px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin-top: 0; color: #555;">종합 투자 예측 지수</h3>
                <h1 style="color: {color}; font-size: 60px; margin: 10px 0;">{pred_val:+.2%}</h1>
                <p style="color: #666; line-height: 1.6;">
                    <b>💡 수치 해석:</b> 글로벌 지표 변화를 종합한 <b>KOSPI 일일 기대 수익률</b>입니다.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("📊 지표별 KOSPI 영향력 비중 (Relative Weight)")
        cont_df = pd.DataFrame(contribution_pct).T
        cont_df.index = ['비중 (%)']
        st.table(cont_df.style.format("{:.1f}%"))

    st.divider()

    # 하단 그래프 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(24, 16))
    plt.subplots_adjust(hspace=0.8, wspace=0.3)

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
        curr_val = plot_data.iloc[-1]
        
        # 임계값 계산
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std

        safe_threshold = threshold if threshold != 0 else 1
        dist = abs(curr_val - threshold) / safe_threshold
        direction = "위로 올라갈 경우" if col in ['Exchange', 'VIX', 'US10Y'] else "아래로 내려갈 경우"
        analysis_text = f"위험선과 약 {dist:.1%} 거리 유지 중\n빨간선 {direction}\n[{warn_text}] 판단"

        # 시각화
        ax.plot(plot_data, color='#34495e', lw=3)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        
        # 날짜 축 가독성 개선 (핵심 수정 부분)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d')) # 연도(2자리)/월/일 축약
        ax.xaxis.set_major_locator(mdates.MaxNLocator(5)) # 눈금 개수를 최대 5개로 제한
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right') # 15도 회전하여 겹침 방지

        ax.text(plot_data.index[5], threshold, f" 산출근거: {th_label}", 
                fontproperties=fprop, fontsize=10, color='#e74c3c', 
                va='bottom', backgroundcolor='#ffffff')

        ax.set_title(title, fontproperties=fprop, fontsize=18, fontweight='bold', pad=15)
        
        # 하단 설명 박스
        ax.text(0.5, -0.4, analysis_text, transform=ax.transAxes, 
                ha='center', va='center', fontproperties=fprop, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.6", fc="#fdfefe", ec="#bdc3c7", lw=1))
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    st.pyplot(fig)

except Exception as e:
    st.error(f"시스템 가동 중 오류 발생: {e}")
