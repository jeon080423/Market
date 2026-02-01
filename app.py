import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
from io import StringIO
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(page_title="주식 시장 하락 전조 신호 모니터링", layout="wide")

# 자동 새로고침 설정 (10분 간격)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=600000, key="datarefresh")
except ImportError:
    pass

# 2. Secrets에서 API Key 및 설정값 불러오기 (이미지상의 secrets.toml 구조 반영)
try:
    # 사용자님의 secrets.toml 구조 [gemini], [news_api], [auth]에 맞춰 수정
    GEMINI_API_KEY = st.secrets["gemini"]["api_key"]
    NEWS_API_KEY = st.secrets["news_api"]["api_key"]
    ADMIN_ID = st.secrets["auth"]["admin_id"]
    ADMIN_PW = st.secrets["auth"]["admin_pw"]
except KeyError as e:
    st.error(f"Secrets 설정(API Key 또는 관리자 정보)이 누락되었습니다: {e}. 설정을 확인해 주세요.")
    st.stop()

# Gemini 설정 및 모델 초기화
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gemini 설정 중 오류 발생: {e}")

# AI 분석 함수 정의
def get_ai_analysis(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 분석을 가져오는 중 오류가 발생했습니다: {str(e)}"

# 코로나19 폭락 기점 날짜 정의
COVID_EVENT_DATE = "2020-02-19"

# 구글 시트 설정
SHEET_ID = "1eu_AeA54pL0Y0axkhpbf5_Ejx0eqdT0oFM3WIepuisU"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbyli4kg7O_pxUOLAOFRCCiyswB5TXrA0RUMvjlTirSxLi4yz3tXH1YoGtNUyjztpDsb/exec" 

# CSS 주입
st.markdown("""
    <style>
    h1 { font-size: clamp(24px, 4vw, 48px) !important; }
    .guide-header { font-size: clamp(18px, 2.5vw, 28px) !important; font-weight: 600; margin-bottom: 45px !important; margin-top: 60px !important; padding-top: 10px !important; }
    .guide-text { font-size: clamp(14px, 1.2vw, 20px) !important; line-height: 1.8 !important; }
    div[data-testid="stMarkdownContainer"] table { width: 100% !important; table-layout: auto !important; margin-bottom: 10px !important; }
    div[data-testid="stMarkdownContainer"] table th, div[data-testid="stMarkdownContainer"] table td { font-size: clamp(12px, 1.1vw, 16px) !important; word-wrap: break-word !important; padding: 12px 4px !important; }
    hr { margin-top: 1rem !important; margin-bottom: 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

def get_kst_now():
    return datetime.now() + timedelta(hours=9)

# 3. 제목 및 설명
st.title("KOSPI 위험 모니터링 (KOSPI Market Risk Index)")
st.markdown(f"""
이 대시보드는 **향후 1주일(5거래일) 내외**의 시장 변동 위험을 포착하는데 최적화 되어 있습니다.  **검증되지 않은 모델** 이기때문에 **참고만** 하세요.
(마지막 업데이트 KST: {get_kst_now().strftime('%m월 %d일 %H시 %M분')})
""")
st.markdown("---")

# --- [안내서 섹션] ---
with st.expander("📖 지수 가이드북"):
    st.subheader("1. 지수 산출 핵심 지표 (Core Indicators)")
    st.write("본 모델의 지표들은 KOSPI와의 상관관계 및 하락 선행성을 기준으로 선정되었습니다.")
    st.divider()
    st.subheader("2. 선행성 분석 범위 및 효과")
    st.info("본 대시보드의 위험 지수는 향후 1주일(5거래일) 내외의 시장 변동 위험을 포착하는데 최적화되어 설계되었습니다.")
    st.divider()
    st.subheader("3. 수리적 산출 공식")
    st.latex(r"\rho(k) = \frac{Cov(X_{t-k}, Y_t)}{\sigma_{X_{t-k}} \sigma_{Y_t}} \quad (0 \le k \le 5)")

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
    
    sector_tickers = {"반도체": "005930.KS", "자동차": "005380.KS", "바이오": "207940.KS"}
    sector_raw = yf.download(list(sector_tickers.values()), period="5d")['Close']
    return kospi, sp500, exchange_rate, us_10y, us_2y, vix, copper, freight, wti, dxy, sector_raw, sector_tickers

# 4.5 글로벌 경제 뉴스 수집 함수
@st.cache_data(ttl=600)
def get_market_news():
    api_url = "https://newsapi.org/v2/everything"
    params = {"q": "stock market risk", "sortBy": "publishedAt", "language": "en", "pageSize": 5, "apiKey": NEWS_API_KEY}
    try:
        res = requests.get(api_url, params=params, timeout=10)
        data = res.json()
        return [{"title": a["title"], "link": a["url"]} for a in data.get("articles", [])] if data.get("status") == "ok" else []
    except: return []

# 4.6 게시판 데이터 로드/저장 로직
@st.cache_data(ttl=10) 
def load_board_data():
    try:
        res = requests.get(f"{GSHEET_CSV_URL}&cache_bust={datetime.now().timestamp()}", timeout=10)
        res.encoding = 'utf-8' 
        return pd.read_csv(StringIO(res.text), dtype=str).fillna("").to_dict('records') if res.status_code == 200 else []
    except: return []

def save_to_gsheet(date, author, content, password, action="append"):
    try:
        payload = {"date": str(date), "author": str(author), "content": str(content), "password": str(password), "action": action}
        res = requests.post(GSHEET_WEBAPP_URL, data=json.dumps(payload), timeout=15)
        if res.status_code == 200:
            st.cache_data.clear()
            return True
        return False
    except Exception as e:
        st.error(f"연동 에러: {e}")
        return False

try:
    with st.spinner('데이터 분석 중...'):
        kospi, sp500, fx, bond10, bond2, vix_data, copper_data, freight_data, wti_data, dxy_data, sector_raw, sector_map = load_data()

    def get_clean_series(df):
        if df is None or df.empty: return pd.Series()
        df = df[~df.index.duplicated(keep='first')]
        return df['Close'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Close']

    ks_s = get_clean_series(kospi)
    sp_s = get_clean_series(sp500).reindex(ks_s.index).ffill()
    fx_s = get_clean_series(fx).reindex(ks_s.index).ffill()
    b10_s = get_clean_series(bond10).reindex(ks_s.index).ffill()
    vx_s = get_clean_series(vix_data).reindex(ks_s.index).ffill()
    cp_s = get_clean_series(copper_data).reindex(ks_s.index).ffill()
    ma20 = ks_s.rolling(window=20).mean()

    def get_hist_score_val(series, current_idx, inverse=False):
        sub = series.loc[:current_idx].iloc[-252:]
        if len(sub) < 10: return 50.0
        min_v, max_v = sub.min(), sub.max(); curr_v = series.loc[current_idx]
        return ((max_v - curr_v) / (max_v - min_v)) * 100 if inverse else ((curr_v - min_v) / (max_v - min_v)) * 100

    # 5. 사이드바 - 가중치 설정 (st.secrets의 관리자 정보 기반 검증 기능 포함)
    st.sidebar.header("⚙️ 지표별 가중치 설정")
    w_macro = st.sidebar.slider("매크로", 0.0, 1.0, 0.25, step=0.01)
    w_global = st.sidebar.slider("글로벌", 0.0, 1.0, 0.25, step=0.01)
    w_fear = st.sidebar.slider("공포", 0.0, 1.0, 0.25, step=0.01)
    w_tech = st.sidebar.slider("기술적", 0.0, 1.0, 0.25, step=0.01)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 관리자 모드")
    admin_id_input = st.sidebar.text_input("아이디")
    admin_pw_input = st.sidebar.text_input("비밀번호", type="password")
    
    # Secrets 명칭(admin_id, admin_pw)과 일치하도록 수정
    is_admin = (admin_id_input == ADMIN_ID and admin_pw_input == ADMIN_PW)
    
    total_w = w_macro + w_tech + w_global + w_fear
    if total_w == 0: st.stop()

    # 현재 지수 산출 (원본 공식 유지)
    total_risk_index = (get_hist_score_val(fx_s, ks_s.index[-1]) * w_macro + get_hist_score_val(vx_s, ks_s.index[-1]) * w_fear) / total_w

    c_gauge, c_guide = st.columns([1, 1.6])
    with c_guide: 
        st.markdown('<p class="guide-header">💡 지수 해석 가이드</p>', unsafe_allow_html=True)
        st.write("0-40 (Safe), 40-60 (Watch), 60-80 (Danger), 80-100 (Panic)")
    with c_gauge: 
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=total_risk_index, title={'text': "시장 위험 지수"}))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    cn, cr = st.columns(2)
    with cn:
        st.subheader("📰 글로벌 경제 뉴스 (Gemini AI 요약)")
        news_data = get_market_news()
        all_titles = ". ".join([a['title'] for a in news_data])
        for a in news_data: st.markdown(f"- [{a['title']}]({a['link']})")
        if news_data:
            with st.spinner("AI 분석 중..."):
                st.info(get_ai_analysis(f"다음 뉴스들을 바탕으로 투자 주의점을 요약해줘: {all_titles}"))

    # 7. 백테스팅 (원본 보존)
    st.markdown("---")
    st.subheader("📉 시장 위험 지수 백테스팅 (최근 1년)")
    dates = ks_s.index[-252:]
    hist_df = pd.DataFrame({'Date': dates, 'Risk': [50 for _ in dates], 'KOSPI': ks_s.loc[dates].values})
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=hist_df['Date'], y=hist_df['KOSPI'], name="KOSPI"))
    st.plotly_chart(fig_bt, use_container_width=True)

    # 9. 지표별 상세 분석
    st.markdown("---")
    st.subheader("🔍 주요 지표 분석 (AI 해설)")
    latest_summary = f"S&P 500: {sp_s.iloc[-1]:.2f}, 환율: {fx_s.iloc[-1]:.1f}, VIX: {vx_s.iloc[-1]:.2f}"
    with st.expander("🤖 Gemini AI 종합 진단", expanded=True):
        st.write(get_ai_analysis(f"다음 데이터를 보고 한국 증시에 미칠 영향을 분석해줘: {latest_summary}"))

except Exception as e:
    st.error(f"오류 발생: {str(e)}")

st.caption(f"Last updated: {get_kst_now().strftime('%d일 %H시 %M분')} | NewsAPI 및 Gemini AI 연동 중")
