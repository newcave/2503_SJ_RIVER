import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Page Title
st.title("Random Forest Prediction GUI with CSV Data")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Checkbox for default data
use_default_data = st.checkbox("Use Default CSV Data")

# Data Loading with encoding fallback
def load_csv(filepath_or_buffer):
    try:
        return pd.read_csv(filepath_or_buffer, encoding='cp949')
    except UnicodeDecodeError:
        return pd.read_csv(filepath_or_buffer, encoding='utf-8')  # fallback

# Data Processing
if use_default_data:
    data_path = "220906_SJ_LSTM_RIVER_TEST002_12cols_200805to15_10min_interval_st104200_31km.csv"
    data = load_csv(data_path)
    st.success("Default CSV data loaded.")
elif uploaded_file:
    data = load_csv(uploaded_file)
    st.success("Uploaded CSV data loaded.")
else:
    st.warning("Please upload a CSV file or select the default data.")
    st.stop()

# Data preprocessing
if 'set_date' in data.columns:
    data = data.drop(['set_date'], axis='columns')

# 주요 지점 선택 체크박스 (모든 컬럼 중에서)
st.subheader("Select Target Points for Prediction")
target_columns = st.multiselect("Select one or more target columns", options=data.columns.tolist(), default=[data.columns[-1]])

if not target_columns:
    st.warning("Please select at least one target column.")
    st.stop()

# 지점별 요약 통계 테이블 누적 저장용
summary_table = []

for target_col in target_columns:
    # 데이터 분할
    feature_cols = [col for col in data.columns if col != target_col]
    train_size = int(len(data) * 0.7)
    trainX, trainY = data[feature_cols][:train_size], data[target_col][:train_size]
    testX, testY = data[feature_cols][train_size:], data[target_col][train_size:]

    # 모델 학습 및 예측
    model = RandomForestRegressor(n_estimators=200, random_state=15)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    # 결과 DataFrame 구성
    result_df = testX.copy()
    result_df['Actual'] = testY.values
    result_df['Prediction'] = predictions

    # 통계 및 메트릭 계산
    stat_data = {
        "location": target_col,
        "Min (Actual)": round(np.min(testY), 3),
        "Max (Actual)": round(np.max(testY), 3),
        "Mean (Actual)": round(np.mean(testY), 3),
        "Variance (Actual)": round(np.var(testY), 3),
        "RMSE": round(np.sqrt(mean_squared_error(testY, predictions)), 3),
        "MSE": round(mean_squared_error(testY, predictions), 3),
        "MAE": round(mean_absolute_error(testY, predictions), 3),
        "R²": round(r2_score(testY, predictions), 3)
    }
    summary_table.append(stat_data)

# 전체 요약 표 출력
if summary_table:
    st.subheader("Summary Table for All Selected Locations")
    summary_df = pd.DataFrame(summary_table)
    summary_df = summary_df.set_index("location")
    st.table(summary_df.round(3))

# 개별 지점별 결과 출력
for target_col in target_columns:
    st.markdown(f"---\n### Results for Target: `{target_col}`")

    # 데이터 분할
    feature_cols = [col for col in data.columns if col != target_col]
    train_size = int(len(data) * 0.7)
    trainX, trainY = data[feature_cols][:train_size], data[target_col][:train_size]
    testX, testY = data[feature_cols][train_size:], data[target_col][train_size:]

    # 모델 재학습 및 예측
    model = RandomForestRegressor(n_estimators=200, random_state=15)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)

    # 결과 DataFrame 구성
    result_df = testX.copy()
    result_df['Actual'] = testY.values
    result_df['Prediction'] = predictions
    st.write(result_df.head(5).round(3))

    # 지점별 요약 통계만 단독 표로도 출력
    st.subheader("Basic Statistics and Metrics")
    single_stat = summary_df.loc[[target_col]].reset_index()
    single_stat.columns.name = "location"
    st.table(single_stat.round(3))

    # 다운로드 버튼
    csv_result = result_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label=f"Download Result CSV ({target_col})",
        data=csv_result,
        file_name=f"predictions_{target_col}.csv",
        mime="text/csv"
    )

    # 시계열 그래프
    st.subheader("Time Series Comparison")
    fig, ax = plt.subplots()
    ax.plot(result_df['Actual'].reset_index(drop=True), label='Actual', color='blue')
    ax.plot(result_df['Prediction'].reset_index(drop=True), label='Predicted', color='orange')
    ax.set_xlabel("Prediction time (Hrs.)")
    ax.set_ylabel("Water Level (EL.m)")
    ax.legend()
    st.pyplot(fig)

    # 산점도
    st.subheader("Scatter Plot of Predicted vs. Actual")
    fig2, ax2 = plt.subplots()
    ax2.scatter(result_df['Actual'], result_df['Prediction'], alpha=0.6, color='green')
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.plot([result_df['Actual'].min(), result_df['Actual'].max()],
             [result_df['Actual'].min(), result_df['Actual'].max()], 'r--', lw=2)
    st.pyplot(fig2)

    # 전체 R² 출력
    overall_r2 = r2_score(result_df['Actual'], result_df['Prediction'])
    st.metric(label=f"Overall R² Score ({target_col})", value=f"{overall_r2:.4f}")
