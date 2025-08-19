import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title("Heart Disease Data Analysis")

uploaded_file = st.file_uploader("Upload your Heart Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Basic Information")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Columns:", df.columns.tolist())

    st.subheader("Statistical Summary")
    st.write(df.describe())

  # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    st.subheader("Correlation Heatmap")
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found for correlation heatmap.")

    # Extra Visualizations
    st.subheader("Gender Distribution")
    if 'sex' in df.columns:
        fig1, ax1 = plt.subplots()
        df['sex'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'], ax=ax1)
        ax1.set_ylabel('')
        st.pyplot(fig1)

    st.subheader("Heart Disease Count")
    if 'target' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='target', data=df, ax=ax2)
        st.pyplot(fig2)

    st.subheader("Age vs Cholesterol")
    if 'age' in df.columns and 'chol' in df.columns:
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x='age', y='chol', hue='target', data=df, ax=ax3)
        st.pyplot(fig3)

    st.subheader("Age Distribution by Heart Disease Status")
    if 'age' in df.columns and 'target' in df.columns:
        fig4, ax4 = plt.subplots()
        sns.boxplot(x='target', y='age', data=df, ax=ax4)
        st.pyplot(fig4)

    # Column Visualization
    st.subheader("Visualize a Column")
    column = st.selectbox("Select column to plot", df.columns)
    fig5, ax5 = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax5)
    st.pyplot(fig5)

else:
    st.info("Please upload a CSV file to start analysis.")

import streamlit as st
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.header("💓 Heart Disease Prediction")

# User Input Fields
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts that you may have Heart Disease.")
    else:
        st.success("✅ The model predicts that you do NOT have Heart Disease.")
