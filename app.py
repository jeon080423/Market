import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [폰트 설정] 로컬 폰트 파일을 직접 로드하는 방식 (가장 확실함)
@st.cache_resource
def set_korean_font():
    # 저장소에 업로드한 NanumGothic.ttf 파일을 직접 로드
    font_file = 'NanumGothic.ttf' 
    
    if os.path.exists(font_file):
        # 폰트 등록
        font_name = fm.FontProperties(fname=font_file).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False # 마이너스 깨짐 방지
        return font_name
    else:
        # 파일이 없을 경우 대비 (Streamlit Cloud의 다른 경로 탐색)
        st.warning("NanumGothic.ttf 파일이 GitHub 저장소에 없습니다. 영문으로 표시됩니다.")
        return None

# 폰트 적용
font_name = set_korean_font()

# [설정] 페이지 설정
st.set_page_config(page_title="KOSPI 위험 지수 분석", layout="wide")

# [데이터 수집] 8대 핵심 지표
@st.cache_data(ttl=3600)
def load_market_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    tickers = {
        '^KS11': 'KOSPI',
        '^SOX': 'SOX',
        '^GSPC': 'SP500',
        '^VIX': 'VIX',
        'USDKRW=X': 'Exchange',
        '^TNX': 'US10Y',
        '^IRX': 'US2Y',
        '000001.SS': 'China'
    }
    
    # 데이터 다운로드
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
    data = data.rename(columns=tickers)
    
    # 전처리
    data = data.ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1) 
    data['Yield_Spread'] = data['US10Y'] - data['US2Y'] 
    
    return data.dropna()

# [회귀 분석]
def perform_analysis(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [메인 화면]
st.title("🛡️ KOSPI 8대 지표 위험 분석 시스템")
st.markdown("글로벌 시장 데이터를 바탕으로 KOSPI의 하락 위험을 통계적으로 진단합니다.")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # 1. 요약 정보
    st.sidebar.subheader(f"📊 모델 설명력: {model.rsquared:.2%}")
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예측 수익률", f"{pred:.2%}")
    with col2:
        status = "위험" if pred < -0.003 else "경계" if pred < 0 else "안정"
        st.subheader(f"시장 진단: {status}")
    with col3:
        st.write(f"최종 업데이트: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # 2. 위험 모니터링 그래프 (한글 폰트 강제 적용 확인)
    st.subheader("⚠️ 주요 지표별 위험 임계점")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # 환율
    axes[0, 0].plot(df['Exchange'].tail(60), color='tab:blue')
    axes[0, 0].axhline(y=1350, color='red', linestyle='--', label='위험(1350)')
    axes[0, 0].set_title("원/달러 환율 (USD/KRW)")
    axes[0, 0].legend()
    
    # VIX
    axes[0, 1].plot(df['VIX'].tail(60), color='tab:purple')
    axes[0, 1].axhline(y=20, color='red', linestyle='--', label='위험(20)')
    axes[0, 1].set_title("공포지수 (VIX)")
    axes[0, 1].legend()
    
    # 반도체 지수 시차
    axes[1, 0].plot(df['SOX_lag1'].tail(60), color='tab:green')
    axes[1, 0].set_title("전일 미 반도체지수 (SOX)")
    
    # 장단기 금리차
    axes[1, 1].plot(df['Yield_Spread'].tail(60), color='tab:orange')
    axes[1, 1].axhline(y=0, color='black')
    axes[1, 1].set_title("장단기 금리차 (10Y-2Y)")

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**분석 가이드:** 환율 1350원과 VIX 20은 지수의 급격한 하락을 유도하는 임계점입니다. SOX(반도체) 지수는 익일 국내 증시의 방향성을 미리 알려주는 핵심 지표입니다.")

except Exception as e:
    st.error(f"데이터 분석 중 오류 발생: {e}")
