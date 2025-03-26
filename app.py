import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Page Title
st.title("Random Forest Prediction GUI with CSV Data")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Checkbox for default data
use_default_data = st.checkbox("Use Default CSV Data")

# Data Processing
if use_default_data:
    data_path = "220906_SJ_LSTM_RIVER_TEST002_12cols_200805to15_10min_interval_st104200_31km.csv"
    data = pd.read_csv(data_path)
    st.success("Default CSV data loaded.")
elif uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Uploaded CSV data loaded.")
else:
    st.warning("Please upload a CSV file or select the default data.")
    st.stop()

# Data preprocessing
if 'set_date' in data.columns:
    data = data.drop(['set_date'], axis='columns')

# Data Preview
st.subheader("Data Preview")
st.write(data.head())

# Dataset Split
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

trainX, trainY = train_data.iloc[:, :-1], train_data.iloc[:, -1]
testX, testY = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Random Forest Model
model = RandomForestRegressor(n_estimators=200, random_state=15)
model.fit(trainX, trainY)

# Predictions
predictions = model.predict(testX)

# Prediction Results
st.subheader("Prediction Results")
result_df = testX.copy()
result_df['Actual'] = testY.values
result_df['Prediction'] = predictions
st.write(result_df.head())

# Download Results
csv_result = result_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Result CSV",
    data=csv_result,
    file_name="predictions.csv",
    mime="text/csv"
)

# Time Series Plot
st.subheader("Time Series Comparison of Actual and Predicted Values")
fig, ax = plt.subplots()
ax.plot(result_df['Actual'].reset_index(drop=True), label='Actual', color='blue')
ax.plot(result_df['Prediction'].reset_index(drop=True), label='Predicted', color='orange')
ax.set_xlabel("Time Steps")
ax.set_ylabel("Values")
ax.legend()
st.pyplot(fig)

# Scatter Plot for Actual vs Predicted
st.subheader("Scatter Plot of Predicted vs. Actual Values")
fig2, ax2 = plt.subplots()
ax2.scatter(result_df['Actual'], result_df['Prediction'], alpha=0.6, color='green')
ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.plot([result_df['Actual'].min(), result_df['Actual'].max()],
         [result_df['Actual'].min(), result_df['Actual'].max()], 'r--', lw=2)
st.pyplot(fig2)

# R-squared for each point
st.subheader("R² Score per Data Point")
r2_scores = [(actual - pred)**2 for actual, pred in zip(result_df['Actual'], result_df['Prediction'])]
r2_df = pd.DataFrame({'R2': r2_scores})
fig3, ax3 = plt.subplots()
ax3.plot(r2_df.index, r2_df['R2'], marker='o', linestyle='', color='purple')
ax3.set_xlabel("Data Points")
ax3.set_ylabel("Squared Errors (lower is better)")
st.pyplot(fig3)

# Overall R² Score
overall_r2 = r2_score(result_df['Actual'], result_df['Prediction'])
st.metric(label="Overall R² Score", value=f"{overall_r2:.4f}")
