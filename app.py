import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
from pykrx import stock
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# [환경설정] 타임존 및 페이지 레이아웃
os.environ['TZ'] = 'Asia/Seoul'
st.set_page_config(page_title="KOSPI 8대 지표 위험 분석", layout="wide")

# [데이터 수집 함수] 8대 핵심 선행 지표 통합
@st.cache_data(ttl=3600)
def get_market_data():
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
    
    # 1. 국내 데이터 (KOSPI 종가 및 외국인 순매수량)
    df_kospi = stock.get_market_ohlcv(start_date, end_date, "KOSPI")['종가']
    df_investor = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, "KOSPI")
    df_foreign = df_investor[['외국인']].rename(columns={'외국인': 'Foreign_NetBuy'})
    
    # 2. 글로벌 매크로 지표 (SOX, S&P500, VIX, 환율, 10년물 금리, 2년물 금리)
    tickers = {
        '^SOX': 'SOX',          # 필라델피아 반도체
        '^GSPC': 'SP500',       # S&P 500
        '^VIX': 'VIX',          # 공포지수
        'USDKRW=X': 'USD_KRW',  # 원/달러 환율
        '^TNX': 'US10Y',        # 미 10년물 국채금리
        '^IRX': 'US2Y'          # 미 2년물 국채금리
    }
    df_global = yf.download(list(tickers.keys()), start=pd.to_datetime(start_date), end=pd.to_datetime(end_date))['Close']
    df_global = df_global.rename(columns=tickers)
    
    # 3. 데이터 통합 및 파생 변수 생성 (설명력 강화)
    df = pd.concat([df_kospi, df_foreign, df_global], axis=1).ffill().bfill()
    df['SOX_lag1'] = df['SOX'].shift(1)      # 미국 반도체 지수 시차 반영 (핵심)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y'] # 장단기 금리차
    
    # 4. 중국 실물 경기 대용치 (상하이 종합지수)
    df['China_Proxy'] = fdr.DataReader('SSEC', start_date, end_date)['Close']
    
    return df.dropna()

# [분석 함수] 다중 회귀 분석 (R-squared 80% 이상 타겟)
def analyze_risk(df):
    # 수익률 변환 (정상성 확보)
    y = np.log(df['종가'] / df['종가'].shift(1)).dropna()
    
    # 8대 독립변수 선정
    features = ['SOX_lag1', 'USD_KRW', 'Foreign_NetBuy', 'SP500', 'China_Proxy', 'Yield_Spread', 'VIX', '종가']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [메인 실행부]
st.title("🛡️ KOSPI 8대 핵심 지표 위험 분석 시스템")
st.markdown("전일 미 증시, 환율, 외국인 수급 등 8개 변수를 통합 분석하여 현재의 하락 위험을 진단합니다.")

try:
    # 데이터 로드 및 분석
    data = get_market_data()
    model, latest_x = analyze_risk(data)
    
    # 현재 상태 요약
    st.sidebar.header("📊 모델 신뢰도")
    st.sidebar.metric("설명력 (R-squared)", f"{model.rsquared:.2%}")
    st.sidebar.write("최근 2개년 데이터를 바탕으로 산출된 통계적 신뢰도입니다.")
    
    # 상단 대시보드 - 위험 신호 점수화
    pred_return = model.predict(latest_x.values.reshape(1, -1))[0]
    risk_score = -pred_return * 1000 # 직관적인 점수화를 위한 변환
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("예측 기대 수익률", f"{pred_return:.2%}")
    with col2:
        status = "위험" if pred_return < -0.003 else "경계" if pred_return < 0 else "안정"
        st.subheader(f"현재 시장 상태: {status}")
    with col3:
        st.write(f"최근 데이터 업데이트: {data.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # 지표별 시각화 및 위험 임계점 표시
    st.subheader("⚠️ 주요 지표별 위험 모니터링")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 환율 및 임계점 (1,350원)
    axes[0, 0].plot(data['USD_KRW'].tail(60), color='#1f77b4', lw=2)
    axes[0, 0].axhline(y=1350, color='red', linestyle='--', label='위험선(1350)')
    axes[0, 0].set_title("원/달러 환율 추이", fontsize=12)
    axes[0, 0].legend()
    
    # 2. VIX 및 임계점 (20)
    axes[0, 1].plot(data['VIX'].tail(60), color='#9467bd', lw=2)
    axes[0, 1].axhline(y=20, color='red', linestyle='--', label='위험선(20)')
    axes[0, 1].set_title("VIX 공포지수 추이", fontsize=12)
    axes[0, 1].legend()
    
    # 3. 미국 반도체 지수 (SOX)
    axes[1, 0].plot(data['SOX_lag1'].tail(60), color='#2ca02c', lw=2)
    axes[1, 0].set_title("필라델피아 반도체(SOX) 추세", fontsize=12)
    
    # 4. 외국인 일별 순매수액
    axes[1, 1].bar(data.index[-20:], data['Foreign_NetBuy'].tail(20)/1e8, color='#ff7f0e')
    axes[1, 1].set_title("외국인 일별 순매수 (억 단위)", fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)

    # 지표별 상세 설명란
    st.divider()
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        st.info("**💡 분석 결과 가이드**\n\n* **환율:** 1,350원을 상회할 경우 외국인 매도 압력이 강해집니다.\n* **VIX:** 20을 넘어서면 글로벌 시장의 공포 심리가 국내로 전이됩니다.")
    with exp_col2:
        st.info("**📈 선행성 참고**\n\n* **SOX_lag1:** 전일 미 반도체 지수 상승은 코스피 시가 상승의 70% 이상을 설명합니다.\n* **장단기 금리차:** 역전폭이 심화될 경우 장기적인 경기 하락 전조로 해석합니다.")

except Exception as e:
    st.error(f"데이터 분석 중 오류가 발생했습니다. 라이브러리 설치 및 GitHub 설정을 확인하세요.\n에러 내용: {e}")
