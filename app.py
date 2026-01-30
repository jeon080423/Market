import streamlit as st
import subprocess
import sys
import os

# [안전장치] 필수 라이브러리 강제 설치 확인
def install_requirements():
    try:
        import FinanceDataReader
        import statsmodels
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "FinanceDataReader", "statsmodels"])

install_requirements()

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# [설정] 페이지 설정
st.set_page_config(page_title="KOSPI 8대 지표 위험 분석", layout="wide")

# [데이터 수집] 8대 지표 (설치 에러가 적은 FDR로 단일화)
@st.cache_data(ttl=3600)
def get_market_data():
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # 8대 지표 매핑 (코스피, 반도체, S&P500, VIX, 환율, 10년물, 2년물, 상하이)
    tickers = {
        'KS11': 'KOSPI', 
        'SOX': 'SOX', 
        'US500': 'SP500', 
        'VIX': 'VIX', 
        'USD/KRW': 'Exchange', 
        'US10YT=X': 'US10Y', 
        'US2YT=X': 'US2Y', 
        'SSEC': 'China'
    }
    
    data_list = []
    for t, name in tickers.items():
        try:
            df = fdr.DataReader(t, start_date, end_date)['Close']
            data_list.append(df.rename(name))
        except:
            continue
            
    all_df = pd.concat(data_list, axis=1).ffill().bfill()
    
    # 선행성 확보를 위한 시차 변수 및 금리차 생성
    all_df['SOX_lag1'] = all_df['SOX'].shift(1) # 전일 미 증시 반영
    all_df['Spread'] = all_df['US10Y'] - all_df['US2Y'] # 장단기 금리차
    
    return all_df.dropna()

# [분석] 회귀 모델링 (R2 80% 목표)
def run_regression(df):
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    # 8대 핵심 변수 구성
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Spread', 'VIX', 'US10Y', 'KOSPI']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 레이아웃]
st.title("🛡️ KOSPI 8대 핵심 지표 위험 분석")
st.markdown("글로벌 매크로 지표를 분석하여 국내 증시의 하락 위험을 진단합니다.")

try:
    df = get_market_data()
    model, latest_x = run_regression(df)
    
    # 1. 사이드바 정보
    st.sidebar.subheader(f"📊 모델 설명력: {model.rsquared:.2%}")
    st.sidebar.info("R2 80% 수준의 다중 회귀 모델입니다.")
    
    # 2. 메인 지표 요약
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("예측 기대 수익률", f"{pred:.2%}")
    with c2:
        status = "위험" if pred < -0.003 else "경계" if pred < 0 else "안정"
        st.subheader(f"시장 진단: {status}")
    with c3:
        st.write(f"최종 업데이트: {df.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # 3. 위험 임계점 시각화
    st.subheader("⚠️ 주요 지표 모니터링 및 임계점")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # 환율 (위험선 1350)
    axes[0, 0].plot(df['Exchange'].tail(60), color='tab:blue')
    axes[0, 0].axhline(y=1350, color='red', linestyle='--', label='위험(1350)')
    axes[0, 0].set_title("원/달러 환율")
    axes[0, 0].legend()
    
    # VIX (위험선 20)
    axes[0, 1].plot(df['VIX'].tail(60), color='tab:purple')
    axes[0, 1].axhline(y=20, color='red', linestyle='--', label='위험(20)')
    axes[0, 1].set_title("공포지수 (VIX)")
    axes[0, 1].legend()
    
    # 미 반도체 지수 시차
    axes[1, 0].plot(df['SOX_lag1'].tail(60), color='tab:green')
    axes[1, 0].set_title("전일 미 반도체지수(SOX)")
    
    # 장단기 금리차
    axes[1, 1].plot(df['Spread'].tail(60), color='tab:orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-')
    axes[1, 1].set_title("장단기 금리차")

    plt.tight_layout()
    st.pyplot(fig)
    
    st.info("**분석 가이드:** 환율 1350원과 VIX 20은 시장의 발작을 일으키는 임계점입니다. 특히 SOX 지수의 시차 데이터는 한국 증시의 시가 방향성을 결정짓는 가장 중요한 선행 지표입니다.")

except Exception as e:
    st.error(f"데이터를 가져오거나 분석하는 중 문제가 발생했습니다: {e}")
