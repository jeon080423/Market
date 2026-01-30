import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. 고정 NewsAPI Key 설정
NEWS_API_KEY = "13cfedc9823541c488732fb27b02fa25"

# 3. 제목 및 설명
st.title("📊 종합 시장 위험 지수(Total Market Risk Index) 모니터링")
st.markdown(f"""
이 대시보드는 상관관계 분석을 통해 **환율(40%), 글로벌(30%), 공포(20%), 기술(10%)** 비중으로 위험 지수를 산출합니다.
(마지막 업데이트: {datetime.now().strftime('%H:%M:%S')})
""")

# 5. 데이터 수집 함수
@st.cache_data(ttl=600)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    nikkei = yf.download("^N225", start=start_date, end=end_date)
    exchange_rate = yf.download("KRW=X", start=start_date, end=end_date)
    us_10y = yf.download("^TNX", start=start_date, end=end_date)
    us_2y = yf.download("^IRX", start=start_date, end=end_date)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    copper = yf.download("HG=F", start=start_date, end=end_date)
    freight = yf.download("BDRY", start=start_date, end=end_date)
    return kospi, sp500, nikkei, exchange_rate, us_10y, us_2y, vix, copper, freight

# 데이터 로드 및 전처리 (중복 제거 및 동기화 강화)
try:
    kospi, sp500, nikkei, fx, bond10, bond2, vix_data, copper_data, freight_data = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        # 중복된 날짜 제거
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

    ks_s, sp_s, nk_s = get_clean_series(kospi), get_clean_series(sp500), get_clean_series(nikkei)
    fx_s, b10_s, b2_s, vx_s = get_clean_series(fx), get_clean_series(bond10), get_clean_series(bond2), get_clean_series(vix_data)
    cp_s, fr_s = get_clean_series(copper_data), get_clean_series(freight_data)

    # 모든 시리즈를 KOSPI 날짜 기준으로 정렬하고 빈 값은 이전 데이터로 채움 (ffill)
    sp_s = sp_s.reindex(ks_s.index).ffill()
    nk_s = nk_s.reindex(ks_s.index).ffill()
    fx_s = fx_s.reindex(ks_s.index).ffill()
    b10_s = b10_s.reindex(ks_s.index).ffill()
    b2_s = b2_s.reindex(ks_s.index).ffill()
    vx_s = vx_s.reindex(ks_s.index).ffill()
    cp_s = cp_s.reindex(ks_s.index).ffill()
    fr_s = fr_s.reindex(ks_s.index).ffill()
    
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    # 가중치 자동 산출 로직
    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max()
            curr_v = series.loc[current_idx]
            if max_v == min_v: return 0.0
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except KeyError: # 날짜가 없을 경우를 대비한 안전장치
            return 50.0

    @st.cache_data(ttl=3600)
    def calculate_sem_weights(_ks_s, _sp_s, _nk_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s):
        lookback = 252
        dates = _ks_s.index[-lookback:]
        data_rows = []
        for d in dates:
            s_sp = get_hist_score_val(_sp_s, d, True)
            s_nk = get_hist_score_val(_nk_s, d, True)
            g_risk = (s_sp * 0.6) + (s_nk * 0.4)
            s_fx = get_hist_score_val(_fx_s, d)
            s_bn = get_hist_score_val(_b10_s, d)
            s_cp = get_hist_score_val(_cp_s, d, True)
            m_score = (s_fx + s_bn + s_cp) / 3
            t_score = max(0, min(100, 100 - (_ks_s.loc[d] / _ma20.loc[d] - 0.9) * 500))
            f_score = get_hist_score_val(_vx_s, d)
            data_rows.append([m_score, g_risk, f_score, t_score, _ks_s.loc[d]])
        df_sem = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'KOSPI'])
        X = df_sem[['Macro', 'Global', 'Fear', 'Tech']]
        X = (X - X.mean()) / X.std()
        Y = (df_sem['KOSPI'] - df_sem['KOSPI'].mean()) / df_sem['KOSPI'].std()
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        abs_coeffs = np.abs(coeffs)
        return abs_coeffs / np.sum(abs_coeffs)

    sem_w = calculate_sem_weights(ks_s, sp_s, nk_s, fx_s, b10_s, cp_s, ma20, vx_s)

    # 4. 사이드바 - 가중치 설정
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, float(round(sem_w[0], 2)), 0.01)
    w_global = st.sidebar.slider("글로벌 시장 위험 (미국/일본)", 0.0, 1.0, float(round(sem_w[1], 2)), 0.01)
    w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, float(round(sem_w[2], 2)), 0.01)
    w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, float(round(sem_w[3], 2)), 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 가중치 산출 근거 (SEM 분석)")
    st.sidebar.write("""
    본 대시보드의 가중치는 **구조방정식(SEM)** 모델링의 다중회귀 분석을 통해 산출되었습니다.
    
    1. **독립변수**: 매크로, 글로벌위험, 시장공포, 기술적지표 점수
    2. **종속변수**: KOSPI 지수
    3. **분석방법**: 최근 252거래일간의 데이터를 시뮬레이션하여 각 지표가 KOSPI 변동에 미치는 **통계적 기여도(표준화 계수)**를 추출했습니다.
    4. **결과반영**: 기여도가 높은 지표에 더 높은 가중치를 부여합니다.
    """)

    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0:
        st.error("가중치의 합이 0일 수 없습니다."); st.stop()

    # 위험 지수 계산
    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        if recent.empty: return 50.0
        min_v, max_v = float(recent.min()), float(recent.max())
        curr_v = float(current_series.iloc[-1])
        if max_v == min_v: return 0.0
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    global_risk_score = (calculate_score(sp_s, sp_s, True) * 0.6) + (calculate_score(nk_s, nk_s, True) * 0.4)
    macro_score = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    tech_score = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = float((macro_score * w_macro + tech_score * w_tech + global_risk_score * w_global + calculate_score(vx_s, vx_s) * w_fear) / total_w)

    # 메인 섹션 레이아웃
    st.markdown("---")
    col_guide, col_gauge = st.columns([1, 1.5])
    with col_guide:
        st.subheader("💡 지수를 더 똑똑하게 보는 법")
        st.markdown("""
        | 점수 구간 | 의미 | 권장 대응 |
        | :--- | :--- | :--- |
        | **0 ~ 40 (Safe)** | 시장 과열 또는 안정기 | 적극적 수익 추구 |
        | **40 ~ 60 (Watch)** | 지표 간 충돌 발생 | 현금 비중 확보 고민 |
        | **60 ~ 80 (Danger)** | 다수 지표 위험 신호 | 공격 투자 지양, 방어적 운용 |
        | **80 ~ 100 (Panic)** | 시스템 위기 가능성 | 리스크 관리 최우선 |
        """)
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = total_risk_index,
            title = {'text': "종합 시장 위험 지수"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"},
                               {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}
        ))
        fig_gauge.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 8. 백테스팅 섹션 (설명 복구 및 에러 해결)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    st.info("""
    **백테스팅(Backtesting)이란?** 과거 데이터를 사용하여 모델이나 투자 전략의 유효성을 검증하는 과정입니다. 
    여기서는 지난 1년간의 데이터를 바탕으로 매일의 '시장 위험 지수'를 재산출하여, 
    실제 KOSPI 지수 하락 시점에 위험 지수가 선행하여 상승했는지 확인합니다.
    """)

    with st.spinner('역사적 데이터 시뮬레이션 중...'):
        lookback = 252
        dates = ks_s.index[-lookback:]
        hist_risks = []
        for d in dates:
            s_sp = get_hist_score_val(sp_s, d, True); s_nk = get_hist_score_val(nk_s, d, True)
            g_risk = (s_sp * 0.6) + (s_nk * 0.4)
            m_score = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
            t_score = max(0, min(100, 100 - (ks_s.loc[d] / ma20.loc[d] - 0.9) * 500))
            total_h = (m_score * w_macro + t_score * w_tech + g_risk * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w
            hist_risks.append(total_h)

        hist_df = pd.DataFrame({'Date': dates, 'RiskIndex': hist_risks, 'KOSPI': ks_s.loc[dates].values})
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['RiskIndex'], name="위험 지수", line=dict(color='red')))
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
        fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(title="KOSPI", overlaying="y", side="right"), height=400)
        st.plotly_chart(fig_bt, use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")
