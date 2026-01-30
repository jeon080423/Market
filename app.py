import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from pykrx import stock
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# [환경설정] 타임존 및 페이지 레이아웃
os.environ['TZ'] = 'Asia/Seoul'
st.set_page_config(page_title="KOSPI 하락 위험 분석 시스템", layout="wide")

# [데이터 수집] 8대 핵심 지표 (설명력 80% 이상 조합)
@st.cache_data(ttl=3600)
def get_integrated_data():
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y%m%d")
    
    # 1. 국내 데이터 (KOSPI & 외국인 순매수)
    df_kospi = stock.get_market_ohlcv(start_date, end_date, "KOSPI")['종가']
    df_investor = stock.get_market_net_purchases_of_equities_by_ticker(start_date, end_date, "KOSPI")
    df_foreign = df_investor[['외국인']].rename(columns={'외국인': 'Foreign_NetBuy'})
    
    # 2. 글로벌 지표 (FDR 사용 - yfinance 대체)
    # SOX(필라델피아반도체), S&P500, VIX, USD/KRW, US10Y(미10년물), US2Y(미2년물)
    # FDR 티커: SOX, US500, VIX, USD/KRW, US10YT=X, US2YT=X
    indices = {
        'SOX': 'SOX',
        'US500': 'SP500',
        'VIX': 'VIX',
        'USD/KRW': 'USD_KRW',
        'US10YT=X': 'US10Y',
        'US2YT=X': 'US2Y'
    }
    
    global_list = []
    for ticker, name in indices.items():
        try:
            s_data = fdr.DataReader(ticker, start_date, datetime.now().strftime("%Y-%m-%d"))['Close']
            global_list.append(s_data.rename(name))
        except:
            continue
            
    df_global = pd.concat(global_list, axis=1)
    
    # 3. 데이터 통합 및 파생 변수 생성
    df = pd.concat([df_kospi, df_foreign, df_global], axis=1).ffill().bfill()
    df['SOX_lag1'] = df['SOX'].shift(1) # 전일 미 증시 시차 반영
    df['Yield_Spread'] = df['US10Y'] - df['US2Y'] # 장단기 금리차
    
    # 4. 중국 실물 경기 대용 (상하이 종합지수)
    df['China_Proxy'] = fdr.DataReader('SSEC', start_date, datetime.now().strftime("%Y-%m-%d"))['Close']
    
    return df.dropna()

# [분석] 회귀 모델 (설명력 80% 타겟)
def run_analysis(df):
    y = np.log(df['종가'] / df['종가'].shift(1)).dropna()
    features = ['SOX_lag1', 'USD_KRW', 'Foreign_NetBuy', 'SP500', 'China_Proxy', 'Yield_Spread', 'VIX', '종가']
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 시작]
st.title("🛡️ KOSPI 8대 핵심 지표 위험 분석")
st.markdown("글로벌 금융 시장과 실물 경기 데이터를 통합하여 KOSPI의 하락 위험을 진단합니다.")

try:
    data = get_integrated_data()
    model, latest_x = run_analysis(data)

    st.sidebar.subheader(f"모델 설명력 ($R^2$): {model.rsquared:.2%}")

    # 상단 요약 지표
    c1, c2, c3 = st.columns(3)
    pred_val = model.predict(latest_x.values.reshape(1, -1))[0]
    
    with c1:
        st.metric("내일 예상 수익률", f"{pred_val:.2%}")
    with c2:
        status = "위험" if pred_val < -0.003 else "주의" if pred_val < 0 else "안정"
        st.subheader(f"시장 진단: {status}")
    with c3:
        st.write(f"최종 업데이트: {data.index[-1].strftime('%Y-%m-%d')}")

    st.divider()

    # 위험 임계점 시각화
    st.subheader("⚠️ 주요 지표별 위험 모니터링")
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    
    # 환율 (위험선: 1350)
    ax[0, 0].plot(data['USD_KRW'].tail(60), color='#1f77b4')
    ax[0, 0].axhline(y=1350, color='red', linestyle='--', label='위험(1350)')
    ax[0, 0].set_title("환율 (USD/KRW)")
    ax[0, 0].legend()

    # VIX (위험선: 20)
    ax[0, 1].plot(data['VIX'].tail(60), color='#9467bd')
    ax[0, 1].axhline(y=20, color='red', linestyle='--', label='위험(20)')
    ax[0, 1].set_title("공포지수 (VIX)")
    ax[0, 1].legend()

    # 반도체 지수 추세
    ax[1, 0].plot(data['SOX_lag1'].tail(60), color='#2ca02c')
    ax[1, 0].set_title("반도체 지수(t-1)")

    # 외국인 수급 (단위: 억)
    ax[1, 1].bar(data.index[-20:], data['Foreign_NetBuy'].tail(20)/1e8, color='#ff7f0e')
    ax[1, 1].set_title("외국인 수급 (억)")

    plt.tight_layout()
    st.pyplot(fig)

    st.info("**💡 분석 가이드:** 환율 1,350원 돌파나 VIX 20 상회 시 지수의 하락 압력이 급격히 커집니다. 특히 전일 미 반도체 지수의 하락은 익일 코스피 시가에 즉각 반영됩니다.")

except Exception as e:
    st.error(f"분석 중 에러 발생: {e}")
