import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# [자동 업데이트] 15분 주기
st_autorefresh(interval=15 * 60 * 1000, key="datarefresh")

# [로컬 데이터 보존 설정]
HISTORY_FILE = 'prediction_history.csv'

def save_prediction_history(date_str, pred_val, actual_close, prev_close):
    """예측 데이터를 로컬 CSV 파일에 저장하여 메모리 유지 (예측 종가, 실제 차이 비교 추가)"""
    pred_close = prev_close * (1 + pred_val)
    diff = actual_close - pred_close # 실제종가 - 예측종가 (오차)
    
    new_data = pd.DataFrame([[
        date_str, 
        f"{pred_val:.4%}", 
        f"{pred_close:,.2f}", 
        f"{actual_close:,.2f}",
        f"{diff:,.2f}", # 종가 차이 추가
        datetime.now().strftime('%H:%M:%S')
    ]], columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])
    
    if os.path.exists(HISTORY_FILE):
        try:
            history_df = pd.read_csv(HISTORY_FILE)
            if date_str not in history_df["날짜"].values:
                current_time = datetime.now().time()
                market_close = datetime.strptime("15:30", "%H:%M").time()
                if current_time >= market_close:
                    history_df = pd.concat([history_df, new_data], ignore_index=True)
                    history_df.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
        except:
            new_data.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')

def load_prediction_history():
    """로컬 CSV 파일에서 히스토리 불러오기"""
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except:
            return pd.DataFrame(columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])
    return pd.DataFrame(columns=["날짜", "전일대비 예측수익률", "예측 종가", "실제 종가", "예측 오차", "기록시각"])

# [폰트 설정]
@st.cache_resource
def get_korean_font():
    font_path = os.path.join(os.getcwd(), 'NanumGothic.ttf')
    if os.path.exists(font_path):
        return fm.FontProperties(fname=font_path)
    return None

fprop = get_korean_font()

st.set_page_config(page_title="KOSPI 인텔리전스 진단 시스템 v3.0", layout="wide")

# [데이터 수집] 개별 수집으로 안정성 확보 및 에러 핸들링 강화
@st.cache_data(ttl=900)
def load_expert_data():
    tickers = {
        '^KS11': 'KOSPI', 'USDKRW=X': 'Exchange', '^SOX': 'SOX', '^GSPC': 'SP500', 
        '^VIX': 'VIX', '000001.SS': 'China', '^TNX': 'US10Y', '^IRX': 'US2Y',
        '005930.KS': 'Samsung', '000660.KS': 'Hynix', '005380.KS': 'Hyundai', '373220.KS': 'LG_Energy'
    }
    start_date = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')
    combined_df = pd.DataFrame()

    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start_date, interval='1d', progress=False)
            if not raw.empty:
                rt = yf.download(ticker, period='1d', interval='1m', progress=False)
                val = rt['Close'].iloc[-1] if not rt.empty else raw['Close'].iloc[-1]
                series = raw['Close'].copy()
                series.iloc[-1] = val
                combined_df[name] = series
        except:
            continue
    
    if combined_df.empty:
        raise Exception("데이터를 불러오지 못했습니다. 네트워크를 확인해주세요.")

    df = combined_df.ffill().interpolate()
    df['SOX_lag1'] = df['SOX'].shift(1)
    df['Yield_Spread'] = df['US10Y'] - df['US2Y']
    return df.dropna().tail(300)

# [분석] 설명력 극대화 모델
def get_analysis(df):
    features_list = ['SOX_lag1', 'Exchange', 'SP500', 'China', 'Yield_Spread', 'VIX', 'US10Y']
    df_smooth = df.rolling(window=3).mean().dropna()
    y = df_smooth['KOSPI']
    X = df_smooth[features_list]
    X_scaled = (X - X.mean()) / X.std()
    X_scaled['SOX_SP500'] = X_scaled['SOX_lag1'] * X_scaled['SP500']
    X_final = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_final).fit()
    
    abs_coeffs = np.abs(model.params.drop(['const', 'SOX_SP500']))
    contribution = (abs_coeffs / abs_coeffs.sum()) * 100
    # 평균과 표준편차를 반환하여 예측 시 동일하게 정규화하도록 함
    return model, contribution, X.mean(), X.std()

def custom_date_formatter(x, pos):
    dt = mdates.num2date(x)
    return dt.strftime('%Y/%m') if dt.month == 1 else dt.strftime('%m')

try:
    df = load_expert_data()
    # 모델 학습 및 통계 정보 추출
    model, contribution_pct, train_mean, train_std = get_analysis(df)
    
    # --- 1. 상단 AI 마켓 브리핑 및 리스크 가이드 ---
    st.title("🏛️ KOSPI 인텔리전스 진단 시스템 v3.0")
    
    # 예측을 위한 데이터 정규화 및 데이터프레임 구성 (에러 해결 핵심)
    features_list = contribution_pct.index.tolist()
    
    def predict_return(target_df):
        # 1. 원본 지표 수준 추출 및 정규화
        level_data = target_df[features_list]
        scaled_data = (level_data - train_mean) / train_std
        
        # 2. 상호작용항 추가
        scaled_data['SOX_SP500'] = scaled_data['SOX_lag1'] * scaled_data['SP500']
        
        # 3. 상수항 추가 및 데이터프레임 순서 정렬
        pred_df = sm.add_constant(scaled_data, has_constant='add')
        
        # 4. 모델 예측
        return model.predict(pred_df).iloc[-1]

    # 단기 예측 (최근 3일 평균 수준 기반)
    current_pred_level = predict_return(df.tail(3).mean().to_frame().T)
    prev_val_level = df['KOSPI'].iloc[-2]
    pred_val = (current_pred_level - prev_val_level) / prev_val_level
    
    # 중기 예측 (최근 20거래일 평균 수준 기반)
    mid_pred_level = predict_return(df.tail(20).mean().to_frame().T)
    mid_start_level = df['KOSPI'].tail(20).iloc[0]
    mid_pred_val = (mid_pred_level - mid_start_level) / mid_start_level

    # 로컬 히스토리 기반 신뢰도 계산
    history_df = load_prediction_history()
    if not history_df.empty:
        history_df['오차수치'] = pd.to_numeric(history_df['예측 오차'].str.replace(',', ''), errors='coerce').abs()
        mae = history_df['오차수치'].tail(5).mean()
        reliability = "높음" if mae < 20 else "보통" if mae < 40 else "주의"
    else:
        reliability = "데이터 부족"

    ai_summary = f"현재 시장은 **{contribution_pct.idxmax()}**의 영향력이 가장 강력하며, 모델 신뢰도는 **{reliability}** 수준입니다."

    header_c1, header_c2 = st.columns([2, 1])
    with header_c1:
        st.info(f"🤖 **AI 마켓 브리핑:** {ai_summary}")
    with header_c2:
        cash_ratio = 10 if pred_val > 0.005 else 30 if pred_val > 0 else 60 if pred_val > -0.005 else 90
        st.metric("추천 현금 비중", f"{cash_ratio}%", delta=f"{'방어' if cash_ratio > 50 else '공격'} 포지션")

    st.divider()

    # --- 2. 기존 기대수익률 및 전략 섹션 ---
    c1, c2, c3 = st.columns([1.1, 1.1, 1.3])
    
    with c1:
        today_str = datetime.now().strftime('%Y-%m-%d')
        save_prediction_history(today_str, pred_val, df['KOSPI'].iloc[-1], prev_val_level)
        color = "#e74c3c" if pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px;">
                <h3 style="margin: 0; color: #555;">📈 KOSPI 기대 수익률: <span style="color:{color}">{pred_val:+.2%}</span></h3>
                <p style="color: #444; font-size: 13px; margin-top: 10px; line-height: 1.5;">
                    <b>[단기 수치 해석]</b><br>
                    8대 지표의 실시간 변화를 다중 회귀 모델에 대입하여 산출한 <b>'KOSPI 기대 수익률'</b>입니다.<br>
                    - <b>(+) 상승 압력 / (-) 하락 압력</b><br>
                    - 절대값이 클수록 글로벌 시장의 에너지가 코스피에 강하게 작용 중임을 의미합니다.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.write("") 
        if not history_df.empty:
            st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; border: 1px solid #eee; background-color: #f9f9f9; max-height: 250px; overflow-y: auto;">
                    <h5 style="margin: 0 0 10px 0;">📊 예측 히스토리</h5>
                    {history_df.tail(10).to_html(index=False, classes='table table-striped')}
                </div>
            """, unsafe_allow_html=True)

    with c2:
        mid_color = "#e74c3c" if mid_pred_val < 0 else "#2ecc71"
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; border-left: 10px solid {mid_color}; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 260px;">
                <h3 style="margin: 0; color: #555;">📅 중기 투자 전망: <span style="color:{mid_color}">{mid_pred_val:+.2%}</span></h3>
                <p style="color: #444; font-size: 13px; margin-top: 10px; line-height: 1.5;">
                    <b>[중기 예측 설명]</b><br>
                    최근 <b>20거래일(약 1개월)</b>간의 글로벌 지표 누적 변화를 바탕으로 산출한 추세적 방향성입니다.<br>
                    - 단기 변동성(Noise)을 제거하고 거시적인 <b>에너지 흐름</b>을 파악하기 위한 지표입니다.<br>
                    - 기대수익률과 방향이 일치할 경우 추세 강화로 해석합니다.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        if pred_val < -0.005 and mid_pred_val < 0:
            signal, s_color = "🔴 즉시 매도", "#ff4b4b"
            reason = "단기 기대수익률이 -0.5%를 하회하며 급락 신호가 발생했고, 중기 추세 에너지 역시 음수(-)로 전환되어 하락 압력이 극에 달한 상태입니다. 리스크 관리를 위해 즉각적인 비중 축소가 권고됩니다."
        elif pred_val < 0:
            signal, s_color = "🟠 매도 준비", "#ffa500"
            reason = "중기 추세는 유지되고 있으나 단기 기대수익률이 음수(-)로 꺾였습니다. 글로벌 지표의 에너지가 약화되고 있으므로 수익 실현을 준비하거나 분할 매도를 검토해야 하는 시점입니다."
        elif pred_val > 0.005 and mid_pred_val > 0:
            signal, s_color = "🔵 매수 유효", "#1f77b4"
            reason = "단기 기대수익률이 +0.5%를 상회하는 강한 반등 신호를 보이고 있으며, 중기 추세 또한 양수(+)로 우상향 에너지가 결합되었습니다. 추세적 상승 가능성이 높은 구간으로 판단됩니다."
        else:
            signal, s_color = "⚪ 보유 및 관망", "#888"
            reason = "단기 변동성과 중기 추세가 혼조세를 보이거나 뚜렷한 방향성을 나타내지 않고 있습니다. 지표가 위험선에 근접할 때까지 추가적인 시장 관망이 필요한 중립 단계입니다."

        sc1, sc2 = st.columns([1.1, 1.4])
        with sc1:
            st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: {s_color}; color: white; text-align: center; height: 140px; display: flex; flex-direction: column; justify-content: center;">
                    <h5 style="margin: 0; font-size: 15px;">⚡ 전략 신호</h5>
                    <h2 style="margin: 5px 0 0 0; font-weight: bold; font-size: 24px;">{signal}</h2>
                </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
                <div style="padding: 12px; border-radius: 10px; border: 1px solid #ddd; background-color: #fff; height: 140px; overflow-y: auto;">
                    <h6 style="margin: 0 0 5px 0; color: #333; font-size: 13px;">🧐 판단 이유</h6>
                    <p style="margin: 0; font-size: 12px; line-height: 1.5; color: #555;">{reason}</p>
                </div>
            """, unsafe_allow_html=True)
        
    with c3:
        st.subheader("🔄 주도 업종 순환매 분석")
        sector_returns = df[['Samsung', 'Hynix', 'Hyundai', 'LG_Energy']].pct_change(5).iloc[-1] * 100
        sector_df = pd.DataFrame(sector_returns).rename(columns={sector_returns.name: '5일 수익률(%)'})
        st.bar_chart(sector_df)
        
        st.subheader("📊 지표별 KOSPI 영향력 비중")
        def highlight_max(s):
            is_max = s == s.max()
            return ['color: red; font-weight: bold' if v else '' for v in is_max]
        cont_df = pd.DataFrame(contribution_pct).T
        st.table(cont_df.style.format("{:.1f}%").apply(highlight_max, axis=1))
        st.caption(f"모델 설명력(R²): {model.rsquared:.2%} | 분석 신뢰도: {reliability}")

    st.divider()

    # --- 3. 8대 지표 그래프 영역 ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    plt.subplots_adjust(hspace=0.4)
    config = [
        ('KOSPI', '1. KOSPI 본체', 'MA250 - 1σ', '선 아래로 하향 시 [추세 붕괴]'),
        ('Exchange', '2. 원/달러 환율', 'MA250 + 1.5σ', '선 위로 상향 시 [외인 자금 이탈]'),
        ('SOX_lag1', '3. 미 반도체(SOX)', 'MA250 - 1σ', '선 아래로 하향 시 [IT 공급망 위기]'),
        ('SP500', '4. 미 S&P 500', 'MA250 - 0.5σ', '선 아래로 하향 시 [글로벌 심리 위축]'),
        ('VIX', '5. 공포지수(VIX)', '20.0', '선 위로 상향 시 [시장 패닉 진입]'),
        ('China', '6. 상하이 종합', 'MA250 - 1.5σ', '선 아래로 하향 시 [아시아권 경기 침체]'),
        ('Yield_Spread', '7. 장단기 금리차', '0.0', '선 아래로 하향 시 [경제 불황 전조]'),
        ('US10Y', '8. 미 국채 10Y', 'MA250 + 1σ', '선 위로 상향 시 [유동성 긴축 압박]')
    ]

    for i, (col, title, th_label, warn_text) in enumerate(config):
        ax = axes[i // 4, i % 4]
        plot_data = df[col].tail(100)
        ma = df[col].rolling(window=250).mean().iloc[-1]
        std = df[col].rolling(window=250).std().iloc[-1]
        if col == 'Exchange': threshold = ma + (1.5 * std)
        elif col in ['VIX', 'Yield_Spread']: threshold = float(th_label)
        elif col in ['US10Y']: threshold = ma + std
        else: threshold = ma - std
        
        ax.plot(plot_data, color='#34495e', lw=2.5)
        ax.axhline(y=threshold, color='#e74c3c', ls='--', lw=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(custom_date_formatter))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.set_title(title, fontproperties=fprop, fontsize=16, fontweight='bold', pad=10)
        ax.text(plot_data.index[0], threshold, f"근거: {th_label}", 
                fontproperties=fprop, color='#e74c3c', va='bottom', fontsize=10, backgroundcolor='#ffffff')
        safe_th = threshold if threshold != 0 else 1
        dist = abs(plot_data.iloc[-1] - threshold) / abs(safe_th)
        ax.set_xlabel(f"위험선 대비 거리: {dist:.1%} | {warn_text}", fontproperties=fprop, fontsize=11, color='#c0392b')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(fprop)

    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"분석 엔진 오류 발생: {e}")
