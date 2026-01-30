import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [보정 로그 저장소] 세션 상태 초기화 (제안 기능)
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

st.set_page_config(page_title="KOSPI 8대 지표 정밀 분석", layout="wide")

# [데이터 수집] 수직 튀기(Spike) 방지 및 로그 기록
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
            ticker_obj = yf.Ticker(t)
            rt_data = ticker_obj.history(period='1d', interval='1m')
            
            if not rt_data.empty and pd.notnull(rt_data['Close'].iloc[-1]):
                val = rt_data['Close'].iloc[-1]
                prev_val = hist_data[t].dropna().iloc[-1]
                
                # 변동성 필터 (10% 이상 급변 시 로그 기록 후 보정)
                diff_pct = (val - prev_val) / prev_val
                if abs(diff_pct) < 0.1:
                    current_prices[t] = val
                else:
                    current_prices[t] = prev_val
                    # 필터링 로그 추가 (제안 기능)
                    log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {tickers[t]} 지표 이상 변동({diff_pct:.2%}) 감지 및 보정 완료"
                    if log_entry not in st.session_state.spike_logs:
                        st.session_state.spike_logs.insert(0, log_entry)
                        st.session_state.spike_logs = st.session_state.spike_logs[:5] # 최근 5개 유지
            else:
                current_prices[t] = hist_data[t].dropna().iloc[-1]
        except:
            current_prices[t] = hist_data[t].dropna().iloc[-1]

    df = hist_data.copy()
    today = pd.Timestamp(datetime.now().date())
    
    if df.index[-1].date() == today.date():
        for t, price in current_prices.items():
            df.at[df.index[-1], t] = price
    else:
        new_row = pd.Series(current_prices, name=pd.Timestamp(datetime.now()))
        df = pd.concat([df, pd.DataFrame([new_row])])

    df = df.rename(columns=tickers).ffill().interpolate(method='linear')
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    
    return df.dropna().tail(250)

# [분석] 회귀 모델링
def perform_analysis(df):
    # 로그 수익률 기반 분석 (이미지와 유사한 영향도 산출용)
    returns = np.log(df / df.shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    
    y = returns['KOSPI']
    X = returns[features]
    
    # 표준화 계수 산출을 위해 데이터 표준화
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = sm.add_constant(X_scaled)
    
    model = sm.OLS(y, X_scaled).fit()
    return model, X_scaled.iloc[-1]

# [UI 구현]
st.title("📊 KOSPI 8대 지표 예측 대시보드")
st.caption(f"최종 갱신: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    df = load_clean_data()
    model, latest_x = perform_analysis(df)
    
    # 예측값 계산
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    if pred < -0.003: s_color, s_icon, s_text = "red", "🚨", "하락 경계"
    elif pred < 0.001: s_color, s_icon, s_text = "orange", "⏳", "중립 / 관망"
    else: s_color, s_icon, s_text = "green", "🚀", "상승 기대"

    st.divider()
    
    # [종합 판단 영역] 첨부 이미지와 유사한 바 차트 추가
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; border: 2px solid {s_color}; text-align: center; background-color: rgba(0,0,0,0.02);">
                <h1 style="font-size: 50px; margin: 0;">{s_icon}</h1>
                <h2 style="color: {s_color}; margin: 5px 0;">{s_text}</h2>
                <p>실시간 예측 수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
        
        # [제안 기능] 데이터 보정 로그 표시
        if st.session_state.spike_logs:
            with st.expander("🔍 실시간 데이터 보정 내역 (최근 5건)"):
                for log in st.session_state.spike_logs:
                    st.caption(log)
        else:
            st.caption("✅ 현재 모든 실시간 데이터가 정상 범위 내에 있습니다.")

    with c2:
        # 지표별 영향도 시각화 (이미지의 분석 결과와 유사한 형태)
        st.subheader("📌 지표별 KOSPI 영향도 (Standardized Beta)")
        coeffs = model.params.drop('const').sort_values()
        
        fig_inf, ax_inf = plt.subplots(figsize=(10, 5))
        colors = ['#ff9999' if x < 0 else '#66b3ff' for x in coeffs]
        coeffs.plot(kind='barh', color=colors, ax=ax_inf)
        
        ax_inf.set_title("각 지표가 오늘 KOSPI에 미치는 상대적 강도", fontproperties=fprop, fontsize=12)
        for label in (ax_inf.get_xticklabels() + ax_inf.get_yticklabels()):
            label.set_fontproperties(fprop)
            
        plt.tight_layout()
        st.pyplot(fig_inf)

    st.divider()

    # 2행 4열 개별 지표 그래프
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    items = [
        ('KOSPI', '1. KOSPI (보정완료)', 'MA250 - 1σ', '평균 대비 저평가'),
        ('Exchange', '2. 환율 (실시간)', 'MA250 + 1.5σ', '급등 경계'),
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
