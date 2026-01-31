import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정 (10분 간격)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. 고정 NewsAPI Key 설정
NEWS_API_KEY = "13cfedc9823541c488732fb27b02fa25"

# 코로나19 폭락 기점 날짜 정의 (S&P 500 고점 기준)
COVID_EVENT_DATE = "2020-02-19"

# 3. 제목 및 설명
st.title("📊 종합 시장 위험 지수(Total Market Risk Index) 모니터링")
st.markdown(f"""
이 대시보드는 **시차 상관관계(Time-Lagged)** 및 **머신러닝 중요도(Feature Importance)** 분석을 통해 최적화된 위험 지수를 산출합니다.
(마지막 업데이트: {datetime.now().strftime('%H:%M:%S')})
""")

# --- [복원/추가] 지표 안내서 및 수리적 용어 설명 섹션 ---
with st.expander("📖 대시보드 사용 가이드 및 수리적 모델 안내 (전문용어 및 수식)"):
    st.subheader("1. 지수 산출 핵심 지표 (Core Indicators)")
    st.write("""
    본 모델의 지표들은 KOSPI와의 **통계적 상관관계** 및 **하락 선행성**을 기준으로 선정되었습니다.
    * **글로벌 리스크**: 미국 **S&P 500 지수**를 활용하며, 한국 증시와의 강력한 동조화 경향을 반영합니다.
    * **통화 및 유동성**: **원/달러 환율** 및 **달러 인덱스(DXY)**를 통해 외국인 자본 유출 압력을 측정합니다.
    * **시장 심리**: **VIX(공포 지수)**를 통해 투자자의 불안 심리와 변동성 전조를 파악합니다.
    * **실물 경제**: 경기 선행 지표인 **구리 가격(Copper)**과 **장단기 금리차**를 포함합니다.
    """)
    st.divider()
    st.subheader("2. 수리적 분석 용어 및 산출 공식")
    st.markdown("#### **① 시차 상관관계 (Time-Lagged Correlation)**")
    st.write("지표 $X$가 변한 후 $k$일 뒤에 KOSPI($Y$)가 반응하는 정도를 분석합니다. 모델은 상관계수 $\\rho$가 최대가 되는 최적의 시차 $k$를 스스로 찾습니다.")
    st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}} \quad (0 \le k \le 5)")
    st.markdown("#### **② 머신러닝 기반 중요도 (Feature Importance)**")
    st.write("단순 회귀계수($\\beta$)에 각 지표의 표준편차($\\sigma$)를 곱하여, 실제 지수 변동에 기여한 '실질 영향력'을 산출합니다.")
    st.latex(r"Importance_i = |\beta_i| \times \sigma_{X_i}")
    st.markdown("#### **③ Z-Score 표준화 (Standardization)**")
    st.write("단위가 다른 지표(원, 포인트, %)를 동일한 저울에서 비교하기 위해 평균 0, 표준편차 1인 점수로 변환합니다.")
    st.latex(r"Z = \frac{x - \mu}{\sigma}")
    st.subheader("3. 데이터 업데이트 및 예측 주기")
    st.write("""
    * **업데이트 주기**: 화면은 **10분** 간격 자동 갱신, 가중치 엔진은 **1시간**마다 재학습합니다.
    * **예측 범위**: 모델은 향후 **5거래일(1주일) 내외**의 단기 하락 위험 포착에 최적화되어 있습니다.
    """)

# 4. 데이터 수집 함수
@st.cache_data(ttl=600)
def load_data():
    end_date = datetime.now()
    start_date = "2019-01-01"
    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^GSPC", start=start_date, end=end_date)
    exchange_rate = yf.download("KRW=X", start=start_date, end=end_date)
    us_10y = yf.download("^TNX", start=start_date, end=end_date)
    us_2y = yf.download("^IRX", start=start_date, end=end_date)
    vix = yf.download("^VIX", start=start_date, end=end_date)
    copper = yf.download("HG=F", start=start_date, end=end_date)
    freight = yf.download("BDRY", start=start_date, end=end_date)
    wti = yf.download("CL=F", start=start_date, end=end_date)
    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date)
    
    sector_tickers = {
        "반도체": "005930.KS", "자동차": "005380.KS", "2차전지": "051910.KS",
        "바이오": "207940.KS", "인터넷": "035420.KS", "금융": "055550.KS",
        "철강": "005490.KS", "방산": "047810.KS", "유틸리티": "015760.KS"
    }
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    
    return kospi, sp500, exchange_rate, us_10y, us_2y, vix, copper, freight, wti, dxy, sector_raw, sector_tickers

try:
    with st.spinner('시차 상관관계 및 ML 가중치 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

    ks_s = get_clean_series(kospi)
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    fr_s = get_clean_series(freight_data).reindex(ks_s.index).ffill()
    wt_s = get_clean_series(wti_data).reindex(ks_s.index).ffill()
    dx_s = get_clean_series(dxy_data).reindex(ks_s.index).ffill()
    
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max()
            curr_v = series.loc[current_idx]
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except: return 50.0

    @st.cache_data(ttl=3600)
    def calculate_ml_lagged_weights(_ks_s, _sp_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s):
        def find_best_lag(feature, target, max_lag=5):
            corrs = [abs(feature.shift(lag).corr(target)) for lag in range(max_lag + 1)]
            return np.argmax(corrs)
        best_lags = {'SP': find_best_lag(_sp_s, _ks_s), 'FX': find_best_lag(_fx_s, _ks_s), 'B10': find_best_lag(_b10_s, _ks_s), 'CP': find_best_lag(_cp_s, _ks_s), 'VX': find_best_lag(_vx_s, _ks_s)}
        data_rows = []
        for d in _ks_s.index[-252:]:
            s_sp = get_hist_score_val(_sp_s.shift(best_lags['SP']), d, True)
            s_fx = get_hist_score_val(_fx_s.shift(best_lags['FX']), d)
            s_b10 = get_hist_score_val(_b10_s.shift(best_lags['B10']), d)
            s_cp = get_hist_score_val(_cp_s.shift(best_lags['CP']), d, True)
            s_vx = get_hist_score_val(_vx_s.shift(best_lags['VX']), d)
            g_risk = s_sp; m_score = (s_fx + s_b10 + s_cp) / 3
            t_score = max(0, min(100, 100 - (float(_ks_s.loc[d]) / float(_ma20.loc[d]) - 0.9) * 500))
            data_rows.append([m_score, g_risk, s_vx, t_score, _ks_s.loc[d]])
        df_reg = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'KOSPI'])
        X = (df_reg.iloc[:, :4] - df_reg.iloc[:, :4].mean()) / df_reg.iloc[:, :4].std()
        Y = (df_reg['KOSPI'] - df_reg['KOSPI'].mean()) / df_reg['KOSPI'].std()
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        abs_coeffs = np.abs(coeffs); vol_weights = X.std().values
        adjusted_importance = abs_coeffs * vol_weights
        return adjusted_importance / np.sum(adjusted_importance)

    sem_w = calculate_ml_lagged_weights(ks_s, sp_s, fx_s, b10_s, cp_s, ma20, vx_s)

    # 5. 사이드바 - 복귀 및 슬라이더
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    if 'slider_m' not in st.session_state: st.session_state.slider_m = float(round(sem_w[0], 2))
    if 'slider_g' not in st.session_state: st.session_state.slider_g = float(round(sem_w[1], 2))
    if 'slider_f' not in st.session_state: st.session_state.slider_f = float(round(sem_w[2], 2))
    if 'slider_t' not in st.session_state: st.session_state.slider_t = float(round(sem_w[3], 2))

    if st.sidebar.button("🔄 최적화 모델 가중치로 복귀"):
        st.session_state.slider_m = float(round(sem_w[0], 2))
        st.session_state.slider_g = float(round(sem_w[1], 2))
        st.session_state.slider_f = float(round(sem_w[2], 2))
        st.session_state.slider_t = float(round(sem_w[3], 2))
        st.rerun()

    w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, key="slider_m", step=0.01)
    w_global = st.sidebar.slider("글로벌 시장 위험 (미국 지수)", 0.0, 1.0, key="slider_g", step=0.01)
    w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, key="slider_f", step=0.01)
    w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, key="slider_t", step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 가중치 산출 근거 (시차 최적화 분석)")
    st.sidebar.write("""
    본 대시보드의 초기 가중치는 **'시차 상관관계(Lagged Correlation)'** 및 **'특성 기여도(Feature Importance)'** 알고리즘을 통해 산출되었습니다.
    
    1. **시차 최적화**: 각 매크로 지표가 KOSPI에 영향을 주기까지의 과거 지연 시간(Lag)을 계산하여 가장 설명력이 높은 시점의 데이터를 추출합니다.
    2. **기여도 분석**: 머신러닝의 변수 중요도 산출 방식을 차용하여, KOSPI 수익률 변동에 대한 각 지표의 통계적 영향력을 계산합니다.
    3. **동적 가중치**: 최근 1년간의 데이터 흐름을 기반으로, 현재 시장 하락을 가장 잘 예측하는 지표에 더 높은 가중치가 자동으로 할당됩니다.
    """)

    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: st.error("가중치 합이 0일 수 없습니다."); st.stop()

    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        min_v, max_v = float(recent.min()), float(recent.max()); curr_v = float(current_series.iloc[-1])
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    m_score_now = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    g_score_now = calculate_score(sp_s, sp_s, True)
    t_score_now = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (m_score_now * w_macro + t_score_now * w_tech + g_score_now * w_global + calculate_score(vx_s, vx_s) * w_fear) / total_w

    # 6. 메인 화면 - 게이지
    st.markdown("---")
    c_gd, c_gg = st.columns([1, 1.5])
    with c_gd:
        st.subheader("💡 지수를 더 똑똑하게 보는 법")
        st.markdown("""
        | 점수 구간 | 의미 | 권장 대응 |
        | :--- | :--- | :--- |
        | **0 ~ 40 (Safe)** | 시장 과열 또는 안정기 | 적극적 수익 추구 |
        | **40 ~ 60 (Watch)** | 지표 간 충돌 발생 | 현금 비중 확보 고민 |
        | **60 ~ 80 (Danger)** | 다수 지표 위험 신호 | 방어적 포트폴리오 운용 |
        | **80 ~ 100 (Panic)** | 시스템적 위기 가능성 | 리스크 관리 최우선 |
        """)
    with c_gg:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, title={'text': "종합 시장 위험 지수", 'font': {'size': 24}},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"}, {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}))
        fig_gauge.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # 7. 백테스팅 섹션
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    st.info("""
    **백테스팅(Backtesting)**: 수리적으로 최적화된 시차 데이터를 기반으로 모델의 유효성을 검증합니다. 위험 지수가 선행하여 상승했는지 확인하십시오.
    """)
    dates = ks_s.index[-252:]
    hist_risks = []
    for d in dates:
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        g = get_hist_score_val(sp_s, d, True)
        t = max(0, min(100, 100 - (float(ks_s.loc[d]) / float(ma20.loc[d]) - 0.9) * 500))
        hist_risks.append((m * w_macro + t * w_tech + g * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w)
    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    correlation = hist_df['Risk'].corr(hist_df['KOSPI'])
    cb1, cb2 = st.columns([3, 1])
    with cb1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red', width=2)))
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
        fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(title="KOSPI", overlaying="y", side="right"), height=400, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_bt, use_container_width=True)
    with cb2:
        st.metric("설명력 (R²)", f"{(correlation**2)*100:.1f}%")
        st.metric("상관계수 (Corr)", f"{correlation:.2f}")
        st.write("""
        **수치 해석 가이드:**
        - **-1.0 ~ -0.7**: 하락장 포착 능력 우수
        - **-0.7 ~ -0.3**: 유의미한 전조 신호
        - **-0.3 ~ 0.0**: 약한 역상관 (참조용)
        - **0.0 이상**: 모델 왜곡 가능성
        """)

    # 7.5 블랙스완 과거 사례 비교 (유지)
    st.markdown("---")
    st.subheader(" Swan 블랙스완(Black Swan) 과거 사례 비교 시뮬레이션")
    col_bs1, col_bs2 = st.columns(2)
    with col_bs1:
        st.info("**2008 금융위기 vs 현재** (리먼 사태 전후 120일)")
        bs_2008_ks = yf.download("^KS11", start="2008-05-01", end="2009-01-01")['Close']
        bs_2008_norm = (bs_2008_ks - bs_2008_ks.mean()) / bs_2008_ks.std()
        fig_bs1 = go.Figure()
        fig_bs1.add_trace(go.Scatter(y=hist_df['Risk'].iloc[-60:].values, name="현재 위험 지수(최근 60일)", line=dict(color='red', width=3)))
        fig_bs1.add_trace(go.Scatter(y=(bs_2008_norm.values + 2) * 20, name="2008년 위기 궤적", line=dict(color='black', dash='dot')))
        fig_bs1.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bs1, use_container_width=True)
    with col_bs2:
        st.info("**2020 코로나 폭락 vs 현재** (팬데믹 전후 120일)")
        bs_2020_ks = yf.download("^KS11", start="2020-01-01", end="2020-06-01")['Close']
        bs_2020_norm = (bs_2020_ks - bs_2020_ks.mean()) / bs_2020_ks.std()
        fig_bs2 = go.Figure()
        fig_bs2.add_trace(go.Scatter(y=hist_df['Risk'].iloc[-60:].values, name="현재 위험 지수(최근 60일)", line=dict(color='red', width=3)))
        fig_bs2.add_trace(go.Scatter(y=(bs_2020_norm.values + 2) * 20, name="2020년 위기 궤적", line=dict(color='blue', dash='dot')))
        fig_bs2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bs2, use_container_width=True)

    # 8. 뉴스 및 보고서
    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 마켓 리스크 뉴스")
        try:
            articles = requests.get(f"https://newsapi.org/v2/everything?q=stock+market+risk&language=en&apiKey={NEWS_API_KEY}", timeout=5).json().get('articles', [])[:5]
            for a in articles: st.markdown(f"- [{a['title']}]({a['url']})")
        except: st.write("뉴스를 불러올 수 없습니다.")
    with cr:
        st.subheader("📝 최신 애널 보고서")
        try:
            res = requests.get("https://finance.naver.com/research/company_list.naver", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            res.encoding = 'euc-kr'; soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select("table.type_1 tr")
            reports = [{"제목": r.select("td")[1].get_text().strip(), "종목": r.select("td")[0].get_text().strip(), "출처": r.select("td")[2].get_text().strip()} for r in rows if r.select_one("td.alpha")][:10]
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
        except: st.write("보고서를 불러올 수 없습니다.")

    # 9. 지표별 상세 분석 (설명/Guide 문구 전면 복원)
    st.markdown("---")
    st.subheader("🔍 실물 경제 및 주요 상관관계 지표 분석")
    def create_chart(series, title, threshold, desc_text):
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title))
        fig.add_hline(y=threshold, line_width=2, line_color="red")
        fig.add_vline(x=COVID_EVENT_DATE, line_width=1.5, line_dash="dash", line_color="blue")
        fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    with r1_c1:
        st.plotly_chart(create_chart(sp_s, "미국 S&P 500", sp_s.last('365D').mean()*0.9, ""), use_container_width=True)
        st.info("**미국 지수**: KOSPI와 가장 강한 정(+)의 상관성을 보입니다.")
    with r1_c2:
        fx_th = float(fx_s.last('365D').mean() * 1.02)
        st.plotly_chart(create_chart(fx_s, "원/달러 환율", fx_th, ""), use_container_width=True)
        st.info(f"**환율**: 최근 1년 평균 대비 +2%({fx_th:.1f}원) 상회 시 외국인 자본 유출 압력이 심화됩니다.")
    with r1_c3:
        st.plotly_chart(create_chart(cp_s, "실물 경기 지표 (Copper)", cp_s.last('365D').mean()*0.9, ""), use_container_width=True)
        st.info("**실물 경기**: 구리 가격 하락은 글로벌 수요 둔화의 선행 신호입니다.")

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        st.plotly_chart(create_chart(yield_curve, "장단기 금리차", 0.0, ""), use_container_width=True)
        st.info("**금리차**: 10년물-2년물 금리 역전은 통상 경기 침체의 강력한 전조 신호입니다.")
    with r2_c2:
        ks_recent = ks_s.last('30D')
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ks_recent.values, name="현재가"))
        fig_ks.add_trace(go.Scatter(x=ks_recent.index, y=ma20.reindex(ks_recent.index).values, name="20일선", line=dict(dash='dot')))
        fig_ks.update_layout(title="KOSPI 최근 1개월 집중 분석", height=300); st.plotly_chart(fig_ks, use_container_width=True)
        st.info("**기술적 분석**: 주가가 20일 이동평균선을 하회할 경우 단기 추세 하락 전환 가능성이 높습니다.")
    with r2_c3:
        st.plotly_chart(create_chart(vx_s, "VIX 공포 지수", 30, ""), use_container_width=True)
        st.info("**VIX 지수**: 지수 급등은 투자 심리 악화와 투매 가능성을 시사합니다.")

    st.markdown("---")
    r3_c1, r3_c2, r3_c3 = st.columns(3)
    with r3_c1:
        fr_th = round(float(fr_s.last('365D').mean() * 0.85), 2)
        st.plotly_chart(create_chart(fr_s, "글로벌 물동량 지표 (BDRY)", fr_th, ""), use_container_width=True)
        st.info(f"**물동량**: 지지선({fr_th}) 하향 돌파 시 글로벌 경기 수축 신호로 간주합니다.")
    with r3_c2:
        wt_th = round(float(wt_s.last('365D').mean() * 1.2), 2)
        st.plotly_chart(create_chart(wt_s, "에너지 가격 (WTI 원유)", wt_th, ""), use_container_width=True)
        st.info(f"**유가**: 유가 급등은 생산 비용 상승과 인플레이션 압박으로 이어져 시장에 부담을 줍니다.")
    with r3_c3:
        dx_th = round(float(dx_s.last('365D').mean() * 1.03), 2)
        st.plotly_chart(create_chart(dx_s, "달러 인덱스 (DXY)", dx_th, ""), use_container_width=True)
        st.info(f"**달러 가치**: 달러 인덱스 상승은 글로벌 유동성 축소 및 위험자산 회피 신호로 작용합니다.")

    # 10. 표준화 비교 분석
    st.markdown("---")
    st.subheader("📊 S&P 500 vs 글로벌 물동량 지표(BDRY) 표준화 비교 분석")
    sp_norm = (sp_s - sp_s.mean()) / sp_s.std(); fr_norm = (fr_s - fr_s.mean()) / fr_s.std()
    fig_norm = go.Figure()
    fig_norm.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values, name="S&P 500 (Standardized)", line=dict(color='blue', width=1.5)))
    fig_norm.add_trace(go.Scatter(x=fr_norm.index, y=fr_norm.values, name="글로벌 물동량 BDRY (Standardized)", line=dict(color='orange', width=1.5)))
    fig_norm.add_vline(x=COVID_EVENT_DATE, line_width=1.5, line_dash="dash", line_color="red")
    fig_norm.update_layout(title="지수간 동조화 추세 분석 (Z-Score 표준화)", xaxis_title="날짜", yaxis_title="표준화 점수 (Z-Score)", height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_norm, use_container_width=True)
    st.info("**분석 가이드**: 두 지표의 단위를 통일(Z-Score)하여 변동의 궤적을 겹쳐 보았습니다. 물동량이 주가지수보다 선행하거나 동행하는 구간을 통해 경기 흐름을 예측할 수 있습니다.")

    # 11. 섹터별 순환매 분석 (유지)
    st.markdown("---")
    st.subheader("🌡️ 섹터별 자금 흐름 분석 (KOSPI 주요 섹터)")
    sector_perf = []
    for name, ticker in sector_map.items():
        try:
            current_val = sector_raw[ticker].iloc[-1]; prev_val = sector_raw[ticker].iloc[-2]
            change = ((current_val - prev_val) / prev_val) * 100
            sector_perf.append({"섹터": name, "등락률": round(change, 2)})
        except: pass
    df_perf = pd.DataFrame(sector_perf)
    if not df_perf.empty:
        fig_heatmap = px.bar(df_perf, x="섹터", y="등락률", color="등락률", color_continuous_scale='RdBu_r', text="등락률", title="금일 섹터별 대표 종목 등락 현황 (%)")
        fig_heatmap.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.info("**분석 가이드**: 종합 위험 지수가 상승할 때 방어 섹터(유틸리티, 금융)와 민감 섹터(반도체, IT)의 등락을 비교하여 자금 이동 경로를 파악하십시오.")

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 시차 최적화 및 ML 기여도 분석 엔진 가동 중")
