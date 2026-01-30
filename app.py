import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# [중요] 라이브러리 체크
try:
    import yfinance as yf
    from pykrx import stock
except ImportError as e:
    st.error(f"라이브러리 로드 실패: {e}")
    st.stop()

# 타임존 및 페이지 설정
os.environ['TZ'] = 'Asia/Seoul'
st.set_page_config(page_title="KOSPI 분석 대시보드", layout="wide")

# [보안] 비밀번호 함수
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if st.session_state["password_correct"]:
        return True
    st.title("🔐 접속 보안")
    password = st.text_input("비밀번호", type="password")
    if st.button("접속"):
        if password == "1234":
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("비밀번호 오류")
    return False

if not check_password():
    st.stop()

# 데이터 수집 및 전처리 로직
@st.cache_data(ttl=3600)
def get_data():
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
        
        # 1. KOSPI 데이터 가져오기 (컬럼명 '종가')
        df_kospi = stock.get_market_ohlcv(start, end, "KOSPI")[['종가']]
        # 문자열일 가능성을 대비해 실수형으로 강제 변환
        df_kospi['종가'] = pd.to_numeric(df_kospi['종가'], errors='coerce')
        
        # 2. 글로벌 지수 데이터 (yfinance)
        tickers = {
            '^SOX': 'SOX', 
            '^GSPC': 'SP500', 
            '^VIX': 'VIX', 
            'USDKRW=X': 'USD_KRW', 
            '^TNX': 'US10Y', 
            '^IRX': 'US2Y'
        }
        df_global = yf.download(list(tickers.keys()), start=pd.to_datetime(start), end=pd.to_datetime(end))['Close']
        df_global = df_global.rename(columns=tickers)
        # 모든 값을 숫자형으로 변환 (에러 발생 시 NaN 처리)
        df_global = df_global.apply(pd.to_numeric, errors='coerce')
        
        # 데이터 병합 (날짜 기준)
        df = pd.concat([df_kospi, df_global], axis=1)
        
        # 3. 결측치 처리 및 파생 변수 생성
        df = df.ffill().bfill()
        df['SOX_lag1'] = df['SOX'].shift(1)
        df['Yield_Spread'] = df['US10Y'] - df['US2Y']
        
        return df.dropna()
    
    except Exception as e:
        st.error(f"데이터 수집 중 상세 오류: {e}")
        return pd.DataFrame()

# 메인 실행부
try:
    data = get_data()
    
    if not data.empty:
        st.success("✅ 데이터 로드 성공!")
        
        # 대시보드 레이아웃
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 KOSPI 종가 추이")
            st.line_chart(data['종가'])
            
        with col2:
            st.subheader("💵 원/달러 환율 추이")
            st.line_chart(data['USD_KRW'])
            
        st.divider()
        st.subheader("📋 최근 데이터 분석 요약 (Last 5 Days)")
        st.dataframe(data.tail())
    else:
        st.warning("수집된 데이터가 없습니다.")

except Exception as e:
    # 이미지에서 발생한 에러를 잡기 위한 예외 처리 강화
    st.error(f"데이터 처리 중 오류 발생: {e}")
    st.info("팁: 데이터 형식 변환 문제일 수 있습니다. pd.to_numeric을 통해 해결을 시도했습니다.")
