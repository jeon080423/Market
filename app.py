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
이 대시보드는 상관관계 분석을 통해 **환율, 글로벌 리스크, 공포지수, 기술적 지표**를 종합하여 위험 지수를 산출합니다.
(마지막 업데이트: {datetime.now().strftime('%H:%M:%S')})
""")

# 4. 데이터 수집 함수
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

# 데이터 로드 및 전처리
try:
    kospi, sp500, nikkei, fx, bond10, bond2, vix_data, copper_data, freight_data = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        df = df[~df.index.duplicated(keep='first')]
        if isinstance(df.columns, pd.MultiIndex): return df['Close'].iloc[:, 0]
        return df['Close']

    ks_s = get_clean_series(kospi)
    # 모든 시리즈를 KOSPI 날짜 기준으로 정렬 및 결측치 채움 (에러 방지 핵심)
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    nk_s = get_clean_series(nikkei).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    b2_s = get_clean_series(bond2).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    fr_s = get_clean_series(freight_data).reindex(ks_s.index).ffill()
    
    yield_curve = b10_s - b2_s
    ma20 = ks_s.rolling(window=20).mean()

    # 가중치 산출 로직
    def get_hist_score_val(series, current_idx, inverse=False):
        try:
            sub = series.loc[:current_idx].iloc[-252:]
            if len(sub) < 10: return 50.0
            min_v, max_v = sub.min(), sub.max()
            curr_v = series.loc[current_idx]
            return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100
        except: return 50.0

    @st.cache_data(ttl=3600)
    def calculate_sem_weights(_ks_s, _sp_s, _nk_s, _fx_s, _b10_s, _cp_s, _ma20, _vx_s):
        dates = _ks_s.index[-252:]
        data_rows = []
        for d in dates:
            s_sp = get_hist_score_val(_sp_s, d, True); s_nk = get_hist_score_val(_nk_s, d, True)
            g_risk = (s_sp * 0.6) + (s_nk * 0.4)
            m_score = (get_hist_score_val(_fx_s, d) + get_hist_score_val(_b10_s, d) + get_hist_score_val(_cp_s, d, True)) / 3
            t_score = max(0, min(100, 100 - (_ks_s.loc[d] / _ma20.loc[d] - 0.9) * 500))
            data_rows.append([m_score, g_risk, get_hist_score_val(_vx_s, d), t_score, _ks_s.loc[d]])
        df_sem = pd.DataFrame(data_rows, columns=['Macro', 'Global', 'Fear', 'Tech', 'KOSPI'])
        X = (df_sem.iloc[:, :4] - df_sem.iloc[:, :4].mean()) / df_sem.iloc[:, :4].std()
        Y = (df_sem['KOSPI'] - df_sem['KOSPI'].mean()) / df_sem['KOSPI'].std()
        coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
        abs_coeffs = np.abs(coeffs)
        return abs_coeffs / np.sum(abs_coeffs)

    sem_w = calculate_sem_weights(ks_s, sp_s, nk_s, fx_s, b10_s, cp_s, ma20, vx_s)

    # 5. 사이드바 설정
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    w_macro = st.sidebar.slider("매크로 (환율/금리/물동량)", 0.0, 1.0, float(round(sem_w[0], 2)), 0.01)
    w_global = st.sidebar.slider("글로벌 시장 위험 (미국/일본)", 0.0, 1.0, float(round(sem_w[1], 2)), 0.01)
    w_fear = st.sidebar.slider("시장 공포 (VIX 지수)", 0.0, 1.0, float(round(sem_w[2], 2)), 0.01)
    w_tech = st.sidebar.slider("국내 기술적 지표 (이동평균선)", 0.0, 1.0, float(round(sem_w[3], 2)), 0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 가중치 산출 근거 (SEM 분석)")
    st.sidebar.write("본 가중치는 **다중회귀분석**을 통해 최근 1년 코스피 변동에 대한 각 지표의 **통계적 기여도**를 추출하여 산출되었습니다.")

    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: st.error("가중치 합이 0일 수 없습니다."); st.stop()

    # 위험 지수 계산
    def calculate_score(current_series, full_series, inverse=False):
        recent = full_series.last('365D')
        min_v, max_v = float(recent.min()), float(recent.max())
        curr_v = float(current_series.iloc[-1])
        return float(max(0, min(100, ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100)))

    macro_score = (calculate_score(fx_s, fx_s) + calculate_score(b10_s, b10_s) + calculate_score(cp_s, cp_s, True)) / 3
    global_score = (calculate_score(sp_s, sp_s, True) * 0.6) + (calculate_score(nk_s, nk_s, True) * 0.4)
    tech_score = max(0.0, min(100.0, float(100 - (float(ks_s.iloc[-1]) / float(ma20.iloc[-1]) - 0.9) * 500)))
    total_risk_index = (macro_score * w_macro + tech_score * w_tech + global_score * w_global + calculate_score(vx_s, vx_s) * w_fear) / total_w

    # 6. 메인 화면: 가이드 및 게이지
    st.markdown("---")
    c_gd, c_gg = st.columns([1, 1.5])
    with c_gd:
        st.subheader("💡 지수 해석 가이드")
        st.markdown("| 점수 | 의미 | 대응 |\n| :--- | :--- | :--- |\n| 0-40 | Safe | 수익 추구 |\n| 40-60 | Watch | 비중 조절 |\n| 60-80 | Danger | 방어 운용 |\n| 80-100 | Panic | 현금 확보 |")
    with c_gg:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, title={'text': "종합 시장 위험 지수"},
            gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 40], 'color': "green"}, {'range': [40, 60], 'color': "yellow"}, {'range': [60, 80], 'color': "orange"}, {'range': [80, 100], 'color': "red"}]}))
        fig_g.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig_g, use_container_width=True)

    # 7. 백테스팅 및 통계 지표 (복원)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    st.info("**백테스팅(Backtesting)**: 과거 데이터를 통해 모델의 유효성을 검증하는 과정입니다. 위험 지수가 선행하여 상승했는지 확인하십시오.")
    
    dates = ks_s.index[-252:]
    hist_risks = []
    for d in dates:
        m = (get_hist_score_val(fx_s, d) + get_hist_score_val(b10_s, d) + get_hist_score_val(cp_s, d, True)) / 3
        g = (get_hist_score_val(sp_s, d, True) * 0.6) + (get_hist_score_val(nk_s, d, True) * 0.4)
        t = max(0, min(100, 100 - (ks_s.loc[d] / ma20.loc[d] - 0.9) * 500))
        hist_risks.append((m * w_macro + t * w_tech + g * w_global + get_hist_score_val(vx_s, d) * w_fear) / total_w)

    hist_df = pd.DataFrame({'Date': dates, 'Risk': hist_risks, 'KOSPI': ks_s.loc[dates].values})
    corr = hist_df['Risk'].corr(hist_df['KOSPI'])
    
    cb1, cb2 = st.columns([3, 1])
    with cb1:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['Risk'], name="위험 지수", line=dict(color='red')))
        fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI", yaxis="y2", line=dict(color='gray', dash='dot')))
        fig_bt.update_layout(yaxis=dict(title="위험 지수", range=[0, 100]), yaxis2=dict(title="KOSPI", overlaying="y", side="right"), height=400)
        st.plotly_chart(fig_bt, use_container_width=True)
    with cb2:
        st.metric("상관계수 (Corr)", f"{corr:.2f}")
        st.metric("설명력 (R²)", f"{(corr**2)*100:.1f}%")
        st.write("음의 상관관계가 강할수록(수치가 낮을수록) 하락장 포착 능력이 우수합니다.")

    # 8. 뉴스 및 보고서 (복원)
    def get_analyst_reports():
        try:
            res = requests.get("https://finance.naver.com/research/company_list.naver", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            res.encoding = 'euc-kr'; soup = BeautifulSoup(res.text, 'html.parser')
            rows = soup.select("table.type_1 tr")
            return [{"제목": r.select("td")[1].get_text().strip(), "종목": r.select("td")[0].get_text().strip(), "출처": r.select("td")[2].get_text().strip()} for r in rows if r.select_one("td.alpha")][:10]
        except: return []

    def get_market_news():
        try: return [{"title": a['title'], "link": a['url']} for a in requests.get(f"https://newsapi.org/v2/everything?q=stock+market+risk&language=en&apiKey={NEWS_API_KEY}", timeout=10).json().get('articles', [])[:5]]
        except: return []

    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 마켓 뉴스")
        for n in get_market_news(): st.markdown(f"- [{n['title']}]({n['link']})")
    with cr:
        st.subheader("📝 최신 리서치 보고서")
        st.dataframe(pd.DataFrame(get_analyst_reports()), use_container_width=True, hide_index=True)

    # 9. 지표별 상세 분석 (복원)
    st.markdown("---")
    st.subheader("🔍 세부 상관관계 지표 분석")
    def create_chart(series, title, threshold, desc):
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name=title))
        fig.add_hline(y=threshold, line_width=2, line_color="red")
        fig.update_layout(title=title, height=280, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    r1_c1, r1_c2, r1_c3 = st.columns(3)
    r1_c1.plotly_chart(create_chart(sp_s, "S&P 500 (미국)", sp_s.last('365D').mean()*0.9, ""), use_container_width=True)
    r1_c2.plotly_chart(create_chart(fx_s, "원/달러 환율", float(fx_s.last('365D').mean()*1.02), ""), use_container_width=True)
    r1_c3.plotly_chart(create_chart(cp_s, "Copper (구리)", cp_s.last('365D').mean()*0.9, ""), use_container_width=True)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    r2_c1.plotly_chart(create_chart(yield_curve, "장단기 금리차", 0.0, ""), use_container_width=True)
    r2_c2.plotly_chart(create_chart(ks_s.last('30D'), "KOSPI 최근 추세", ma20.iloc[-1], ""), use_container_width=True)
    r2_c3.plotly_chart(create_chart(vx_s, "VIX 공포 지수", 30, ""), use_container_width=True)

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | SEM 분석 및 결측치 보정 완료")
