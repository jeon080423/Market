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

st.set_page_config(page_title="KOSPI 정밀 진단 시스템 v2.5", layout="wide")

# [데이터 수집] 멀티인덱스 충돌을 방지하기 위한 개별 수집 로직
@st.cache_data(ttl=300)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y'
    }
    
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    combined_df = pd.DataFrame()

    # 에러 방지를 위해 지표별로 안전하게 개별 다운로드
    for ticker, name in tickers.items():
        try:
            # 과거 데이터 + 최신 장중 데이터 포함 다운로드
            raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
            if not raw.empty:
                series = raw['Close'].copy()
                # 최신 장중가(1분봉) 확인 및 업데이트
                rt = yf.download(ticker, period='1d', interval='1m', progress=False)
                if not rt.empty:
                    series.iloc[-1] = rt['Close'].iloc[-1]
                combined_df[name] = series
        except:
            continue

    # 데이터 보정 및 파생 변수 생성
    df = combined_df.ffill().interpolate(method='linear')
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
st.caption(f"최종 업데이트: {datetime.now().strftime('%y/%m/%d %H:%M:%S')} (5분 자동 갱신)")

try:
    df = load_expert_data()
    model, contribution_pct = get_analysis(df)
    
    # --- 1. 상단 요약 영역 ---
    c1, c2 = st.columns([1, 1.5])
    with c1:
        current_chg = (df.iloc[-1] / df.iloc[-2] - 1)
        pred_input = [1] + [current_chg[f] for f in contribution_pct.index]
        pred_val = model.predict(pred_input)[0]
        
        color = "#e74c3c" if pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 25px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; color: #555;">종합 투자 예측 지수</h3>
                <h1 style="color: {color}; font-size: 50px; margin: 10px 0;">{pred_val:+.2%}</h1>
                <p style="color: #666; font-size: 14px; line-height: 1.5;">
                    <b>💡 해석:</b> 8대 글로벌 지표를 복합 분석한 <b>KOSPI 일일 기대 수익률</b>입니다. 수치가 양(+)일수록 지표 환경이 우호적임을 의미합니다.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.subheader("📊 지표별 KOSPI 영향력 비중 (Relative Weight)")
        cont_df = pd.DataFrame(contribution_pct).T
        cont_df.index = ['비중 (%)']
        st.table(cont_df.style.format("{:.1f}%"))
        st.caption("※ 산출 근거: 표준화 회귀 계수(Beta) 절대값 기반 비중 환산")

    st.divider()

    # --- 2. 하단 8대 지표 그래프 (2행 4열) ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 16))
    plt.subplots_adjust(hspace=0.9, wspace=0.3)

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
        
        # 위험선 계산
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std

        # 진단 가이드 텍스트
        safe_th = threshold if threshold != 0 else 1e-6
        dist = abs(curr_val - threshold) / abs(safe_th)
        direction = "위로 상향 돌파 시" if col in ['Exchange', 'VIX', 'US10Y'] else "아래로 하향 이탈 시"
        analysis_text = f"위험선과 약 {dist:.1%} 거리 유지 중\n지수가 빨간선 {direction}\n[{warn_text}] 상태로 판단"

        # 시각화
        ax.plot(plot_data, color='#34495e', lw=3)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        
        # 위험선 산출 근거 표기 (선 근처)
        ax.text(plot_data.index[5], threshold, f" 산출근거: {th_label}", 
                fontproperties=fprop, fontsize=10, color='#e74c3c', 
                va='bottom', backgroundcolor='#ffffff', alpha=0.9)

        # 가로축 날짜 최적화
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m/%d'))
        ax.xaxis.set_major_locator(mdates.MaxNLocator(5))
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontproperties=fprop)

        ax.set_title(title, fontproperties=fprop, fontsize=18, fontweight='bold', pad=15)
        
        # 하단 전문 진단 박스 (겹침 방지 간격 최적화)
        ax.text(0.5, -0.45, analysis_text, transform=ax.transAxes, 
                ha='center', va='center', fontproperties=fprop, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.6", fc="#fdfefe", ec="#bdc3c7", lw=1))
        
        for label in (ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    st.pyplot(fig)

except Exception as e:
    st.error(f"시스템 가동 중 오류 발생: {e}")
