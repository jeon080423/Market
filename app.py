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

# [세션 상태] 보정 로그 및 데이터 유지
if 'spike_logs' not in st.session_state:
    st.session_state.spike_logs = []

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 8대 지표 정밀 진단", layout="wide")

# [데이터 수집] 실시간 보정 및 로그 기록 로직
@st.cache_data(ttl=300)
def load_clean_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d', progress=False)['Close']
    
    current_prices = {}
    for t in tickers.keys():
        try:
            # 실시간 데이터 추출
            ticker_obj = yf.Ticker(t)
            rt_data = ticker_obj.history(period='1d', interval='1m')
            
            if not rt_data.empty:
                val = rt_data['Close'].iloc[-1]
                prev_val = hist_data[t].dropna().iloc[-1]
                
                # 변동성 필터링 (10% 이상 이상치)
                diff_pct = (val - prev_val) / prev_val
                if abs(diff_pct) < 0.1:
                    current_prices[t] = val
                else:
                    current_prices[t] = prev_val
                    log_msg = f"{datetime.now().strftime('%H:%M:%S')} | {tickers[t]} 보정 완료 ({diff_pct:.1%})"
                    if log_msg not in st.session_state.spike_logs:
                        st.session_state.spike_logs.insert(0, log_msg)
            else:
                current_prices[t] = hist_data[t].dropna().iloc[-1]
        except:
            current_prices[t] = hist_data[t].dropna().iloc[-1]

    # 데이터 병합
    df = hist_data.copy()
    today_ts = pd.Timestamp(datetime.now().date())
    
    if df.index[-1].date() == today_ts.date():
        for t, price in current_prices.items():
            df.at[df.index[-1], t] = price
    else:
        new_row = pd.Series(current_prices, name=pd.Timestamp(datetime.now()))
        df = pd.concat([df, pd.DataFrame([new_row])])

    df = df.rename(columns=tickers).ffill().interpolate(method='linear')
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    
    return df.dropna().tail(250)

# [분석] 회귀 모델링 (영향도 산출용 표준화 회귀)
def perform_analysis(df):
    returns = np.log(df / df.shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    
    y = returns['KOSPI']
    X = returns[features]
    
    # 계수 비교를 위한 표준화 (Z-score)
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = sm.add_constant(X_scaled)
    
    model = sm.OLS(y, X_scaled).fit()
    return model, X_scaled.iloc[-1]

# [UI 구현]
st.title("🛡️ KOSPI 8대 지표 예측 및 실시간 진단")
st.caption(f"최종 업데이트: {datetime.now().strftime('%H:%M:%S')} (5분 자동 갱신)")

try:
    df = load_clean_data()
    model, latest_x = perform_analysis(df)
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    # 종합 예측 상태 설정
    if pred < -0.003: s_color, s_icon, s_text = "red", "🚨", "하락 경계"
    elif pred < 0.001: s_color, s_icon, s_text = "orange", "⏳", "중립 / 관망"
    else: s_color, s_icon, s_text = "green", "🚀", "상승 기대"

    st.divider()
    
    # --- 상단 종합 판단 영역 ---
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {s_color}; text-align: center; background-color: rgba(0,0,0,0.02);">
                <h1 style="font-size: 50px; margin: 0;">{s_icon}</h1>
                <h2 style="color: {s_color}; margin: 10px 0;">{s_text}</h2>
                <p>예측 기대 수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.spike_logs:
            with st.expander("🔍 최근 데이터 보정 내역"):
                for log in st.session_state.spike_logs[:5]:
                    st.caption(log)
        else:
            st.caption("✅ 모든 실시간 데이터가 정상 범위입니다.")

    with c2:
        # 지표별 영향도 그래프 (첨부 이미지 스타일)
        st.subheader("📌 지표별 KOSPI 영향도 (Standardized Beta)")
        coeffs = model.params.drop('const').sort_values()
        
        fig_inf, ax_inf = plt.subplots(figsize=(10, 5))
        colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in coeffs]
        bars = ax_inf.barh(coeffs.index, coeffs.values, color=colors)
        
        ax_inf.axvline(0, color='black', lw=1)
        ax_inf.set_title("각 지표가 오늘 코스피 방향에 주는 영향력", fontproperties=fprop, fontsize=12)
        
        for label in (ax_inf.get_xticklabels() + ax_inf.get_yticklabels()):
            label.set_fontproperties(fprop)
            
        plt.tight_layout()
        st.pyplot(fig_inf)

    st.divider()

    # --- 하단 8대 지표 그래프 ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    items = [
        ('KOSPI', '1. KOSPI', 'MA250 - 1σ', '저평가 구간'),
        ('Exchange', '2. 환율', 'MA250 + 1.5σ', '급등 경계'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '단기 저점'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '추세 주의'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (Fix)', '패닉 구간'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '경기 침체'),
        ('Yield_Spread', '7. 금리차', '0.00 (Fix)', '불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '금리 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(items):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(60)
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = 20.0 if col == 'VIX' else 0.0
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='#1f77b4', lw=3)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.9, lw=2)
        ax.text(plot_data.index[2], threshold, f" {threshold_label}", 
                fontproperties=fprop, fontsize=11, color='crimson', 
                verticalalignment='bottom', backgroundcolor='white')

        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=15)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)
        ax.annotate(f"[{desc}]", xy=(0.5, -0.18), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='#444444')

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"대시보드 실행 중 오류 발생: {e}")
    st.info("데이터를 다시 불러오는 중입니다. 5분마다 자동 갱신됩니다.")
