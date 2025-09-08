import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide"
)

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

df, data = load_data()

# ----------------------------
# Train model
# ----------------------------
@st.cache_resource
def train_model():
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ----------------------------
# Sidebar Info
# ----------------------------
st.sidebar.title("ℹ️ How to Use")
st.sidebar.info("""
- You can **manually enter values** in the sidebar  
- Or **upload a CSV file** with the same columns as the dataset  
- The model predicts if the tumor is **Benign (0)** or **Malignant (1)**  
- Use **Download Results** to save predictions as a CSV.
""")

st.sidebar.header("Manual Input Features")
input_data = {}
for feature in data.feature_names:
    input_data[feature] = st.sidebar.number_input(
        feature,
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )
manual_input_df = pd.DataFrame([input_data])

# ----------------------------
# Main Layout
# ----------------------------
st.title("🎗️ Breast Cancer Prediction App")
st.markdown("This application helps predict whether a tumor is **Benign** or **Malignant** using patient diagnostic data.")

# Two-column layout for upload + preview
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            # Standardize column names to match sklearn dataset
            input_df.columns = [col.replace("_mean", " mean")
                                   .replace("_se", " error")
                                   .replace("_worst", " worst")
                                   .replace("_", " ")
                                for col in input_df.columns]

            # Keep only the features used in training
            input_df = input_df[data.feature_names]

            st.success("✅ File uploaded and processed successfully!")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            input_df = manual_input_df
    else:
        input_df = manual_input_df


with col2:
    st.subheader("📊 Example Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ----------------------------
# Prediction Section
# ----------------------------
st.markdown("---")
st.subheader("🔮 Prediction Results")

predictions = model.predict(input_df)
probabilities = model.predict_proba(input_df)

results_df = input_df.copy()
results_df["Prediction"] = predictions
results_df["Prediction"] = results_df["Prediction"].map({0: "Benign", 1: "Malignant"})

st.write(results_df)

# Show single case result if only 1 row
if len(results_df) == 1:
    label = results_df["Prediction"].values[0]
    confidence = np.max(probabilities) * 100
    if label == "Benign":
        st.success(f"✅ The tumor is **{label}** with **{confidence:.2f}% confidence**.")
    else:
        st.error(f"⚠️ The tumor is **{label}** with **{confidence:.2f}% confidence**.")
else:
    st.info("Multiple rows detected. See the table above for predictions.")

# ----------------------------
# Download Option
# ----------------------------
csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name="breast_cancer_predictions.csv",
    mime="text/csv",
)
