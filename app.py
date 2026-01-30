import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import os

# [자동 업데이트] 5분마다 새로고침 (페이지 최상단 배치)
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

# [설정] 페이지 설정
st.set_page_config(page_title="KOSPI 8대 지표 실시간 예측", layout="wide")

# [데이터 수집] 실시간 데이터 반영 및 결합 안정화
@st.cache_data(ttl=300)
def load_market_data():
    tickers = {
        '^KS11': 'KOSPI', '^SOX': 'SOX', '^GSPC': 'SP500', '^VIX': 'VIX',
        'USDKRW=X': 'Exchange', '^TNX': 'US10Y', '^IRX': 'US2Y', '000001.SS': 'China'
    }
    
    # 1. 과거 일봉 데이터 수집
    start_date = (datetime.now() - timedelta(days=1000)).strftime('%Y-%m-%d')
    hist_data = yf.download(list(tickers.keys()), start=start_date, interval='1d')['Close']
    
    # 2. 실시간 데이터 가져오기
    current_data = {}
    for t in tickers.keys():
        try:
            # 장중 1분봉 데이터의 마지막 값을 가져옴
            tmp = yf.download(t, period='1d', interval='1m', progress=False)
            if not tmp.empty:
                current_data[t] = tmp['Close'].iloc[-1]
            else:
                current_data[t] = hist_data[t].iloc[-1]
        except:
            current_data[t] = hist_data[t].iloc[-1]

    # 3. 데이터 결합 (인덱스 에러 방지)
    hist_df = hist_data.copy()
    last_date = hist_df.index[-1]
    
    # 만약 오늘 데이터가 이미 hist_df에 있다면 업데이트, 없다면 새로 추가
    today_date = pd.Timestamp(datetime.now().date())
    
    # 데이터가 Series인 경우와 DataFrame인 경우 모두 대응 가능하게 처리
    temp_row = pd.Series(current_data)
    
    if last_date.date() >= today_date.date():
        hist_df.iloc[-1] = temp_row
    else:
        # 새로운 행 추가
        new_row_df = pd.DataFrame([temp_row], index=[pd.Timestamp(datetime.now())])
        hist_df = pd.concat([hist_df, new_row_df])
    
    data = hist_df.rename(columns=tickers).ffill().bfill()
    data['SOX_lag1'] = data['SOX'].shift(1)
    data['Yield_Spread'] = data['US10Y'] - data['US2Y']
    
    return data.dropna()

# [분석] 회귀 모델링
def perform_analysis(df):
    # 수익률 계산
    y = np.log(df['KOSPI'] / df['KOSPI'].shift(1)).dropna()
    features = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y', 'KOSPI']
    
    # 분석에 필요한 컬럼이 모두 있는지 확인
    X = df[features].pct_change().loc[y.index].replace([np.inf, -np.inf], 0).fillna(0)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit()
    return model, X.iloc[-1]

# [UI 구현]
st.title("📊 KOSPI 8대 지표 실시간 예측 및 투자 가이드")
st.caption(f"최근 데이터 확인 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (5분마다 자동 갱신)")

try:
    df = load_market_data()
    model, latest_x = perform_analysis(df)
    
    # 예측 신호 산출
    pred = model.predict(latest_x.values.reshape(1, -1))[0]
    
    # 신호 판정
    if pred < -0.003:
        s_color, s_icon, s_text = "red", "🚨", "하락 경계 (Risk Off)"
        s_guide = "시장 지표가 부정적입니다. 현금 비중을 확대하고 방어적으로 대응하세요."
    elif pred < 0.001:
        s_color, s_icon, s_text = "orange", "⏳", "중립 (Neutral / Watch)"
        s_guide = "상/하방 에너지가 균형을 이루고 있습니다. 무리한 매매보다는 추세 확인을 권장합니다."
    else:
        s_color, s_icon, s_text = "green", "🚀", "상승 기대 (Risk On)"
        s_guide = "글로벌 지표가 긍정적입니다. 주도주 중심의 매수 관점 접근이 유리합니다."

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; border: 2px solid {s_color}; background-color: rgba(0,0,0,0.05); text-align: center;">
                <h1 style="font-size: 60px; margin: 0;">{s_icon}</h1>
                <h2 style="color: {s_color}; margin: 10px 0;">{s_text}</h2>
                <p style="font-size: 18px;">실시간 기대수익률: <b>{pred:.2%}</b></p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.subheader("💡 실시간 투자 행동 가이드")
        st.info(s_guide)
        st.write(f"**통계적 신뢰도:** 설명력(R²) **{model.rsquared:.2%}** | 8개 지표의 실시간 변화를 복합 분석한 결과입니다.")

    st.divider()

    # 지표 그래프 (2행 4열)
    fig, axes = plt.subplots(2, 4, figsize=(24, 13))
    plt.rcParams['axes.unicode_minus'] = False

    items = [
        ('KOSPI', '1. KOSPI (실시간)', 'MA250 - 1σ', '평균 대비 저평가'),
        ('Exchange', '2. 환율 (실시간)', 'MA250 + 1.5σ', '급등 경계'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', 'AI 업황 저점'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '추세 훼손 주의'),
        ('VIX', '5. 공포지수(VIX)', '20.0 (Fixed)', '패닉 임계점'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '중국 경기 침체'),
        ('Yield_Spread', '7. 장단기 금리차', '0.00 (Fixed)', '불황 전조'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '금리 압박')
    ]

    for i, (col, title, threshold_label, desc) in enumerate(items):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(120)
        ma250 = df[col].rolling(window=250).mean().iloc[-1]
        std250 = df[col].rolling(window=250).std().iloc[-1]
        
        if col == 'Exchange': threshold = ma250 + (1.5 * std250)
        elif col in ['VIX', 'Yield_Spread']: threshold = 20.0 if col == 'VIX' else 0.0
        elif col in ['US10Y']: threshold = ma250 + std250
        else: threshold = ma250 - std250
        
        ax.plot(plot_data, color='#1f77b4', lw=2.5)
        ax.axhline(y=threshold, color='crimson', linestyle='--', alpha=0.9, lw=2)
        ax.text(plot_data.index[5], threshold, f" 위험 기준: {threshold_label}", 
                fontproperties=fprop, fontsize=11, color='crimson', 
                verticalalignment='bottom', backgroundcolor='white')

        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=15)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)
        ax.annotate(f"[{desc}]", xy=(0.5, -0.18), xycoords='axes fraction', 
                    ha='center', fontproperties=fprop, fontsize=12, color='#444444')

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"데이터 연동 중 오류 발생: {e}")
    st.info("데이터를 불러오는 중입니다. 잠시만 기다려 주세요.")
