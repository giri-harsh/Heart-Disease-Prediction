import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Load dataset
#heart_data = pd.read_csv(r'C:\Users\Harsh Giri\OneDrive\Documents\Programing Language\Python\Project\Disease prediction\Heart Disease\heart.csv')


st.title("Heart Disease Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    heart = pd.read_csv(uploaded_file)
    st.write(heart.head())  # Display the first few rows


# Prepare data
X = heart.drop(columns='target', axis=1)
Y = heart['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# App Title
st.title("Heart Disease Prediction AI 🏥")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.radio("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])

# Resting Blood Pressure Input (Slider + Number)
col1, col2 = st.columns([3, 1])  
with col1:
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, value=120, step=1)

with col2:
    trestbps_input = st.number_input("", min_value=80, max_value=200, value=trestbps, step=1)

# Keep both inputs in sync
trestbps = trestbps_input

# Cholesterol Input (Slider + Number)
col1, col2 = st.columns([3, 1])  
with col1:
    chol = st.slider("Cholesterol Level (mg/dL)", 100, 600, value=200, step=1)

with col2:
    chol_input = st.number_input("", min_value=100, max_value=600, value=chol, step=1)

chol = chol_input

# Other Inputs
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", options=["Yes", "No"])
restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
# thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, step=1)
# Maximum Heart Rate Achieved (Slider + Number Input)
col1, col2 = st.columns([3, 1])  
with col1:
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, value=150, step=1)

with col2:
    thalach_input = st.number_input("", min_value=60, max_value=220, value=thalach, step=1)

thalach = thalach_input  # Keep both inputs in sync

exang = st.radio("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# Fix: Ensure `trestbps` and `chol` are properly assigned
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# Scale input
input_data_scaled = scaler.transform(input_data)

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.success("✅ The Person **does NOT** have Heart Disease.")
    else:
        st.error("⚠️ The Person **HAS** Heart Disease. Please consult a doctor.")
