import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Cancer Predictor Pro",
    page_icon="♋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads and prepares data from the CSV file."""
    df = pd.read_csv(file_path)
    df = df.drop(columns=['id'], axis=1)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    df = df.drop(columns=['diagnosis'], axis=1)
    feature_names = df.columns.drop('target').tolist()
    target_names = ['Benign', 'Malignant']
    return df, feature_names, target_names

try:
    df, feature_names, target_names = load_data('breast-cancer.csv')
except FileNotFoundError:
    st.error("Error: 'breast-cancer.csv' not found. Please place it in the same directory.")
    st.stop()

# --- Model Training ---
@st.cache_resource
def train_model():
    """Trains and returns the Random Forest model."""
    X = df[feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

# --- Sidebar for User Input ---
with st.sidebar:
    # Adding a logo or an image to the sidebar
    st.image("https://www.freeiconspng.com/uploads/cancer-ribbon-png-10.png", width=100)
    st.title("Tumor Feature Input")
    st.markdown("Adjust the sliders to match the tumor's measurements.")
    
    input_df = pd.DataFrame(columns=feature_names)
    for feature in feature_names:
        input_df.loc[0, feature] = st.slider(
            label=feature.replace('_', ' ').title(),
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean())
        )
    st.caption("Sliders are preset to the dataset's average values.")


# --- Main Page Content ---
st.title("Breast Cancer Predictor Pro")
st.markdown("An interactive tool to predict breast cancer using a machine learning model.")

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["🎯 **Prediction**", "📊 **Model Performance**", "📈 **Feature Analysis**"])

# --- Prediction Tab ---
with tab1:
    st.header("Prediction Result")
    st.markdown("Based on the features you provided in the sidebar, the model predicts:")

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    is_malignant = (prediction[0] == 1)
    prediction_label = "Malignant" if is_malignant else "Benign"
    confidence_score = prediction_proba[0][prediction[0]]

    # Layout for prediction and confidence
    col1, col2 = st.columns(2)
    with col1:
        if is_malignant:
            st.warning(f"## {prediction_label}")
        else:
            st.success(f"## {prediction_label}")
        st.caption("This is the model's prediction for the given tumor features.")

    with col2:
        st.metric(label="Confidence Score", value=f"{confidence_score:.2%}")
        st.caption("This score represents the model's confidence in its prediction.")
    
    st.info("ℹ️ **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.")


# --- Model Performance Tab ---
with tab2:
    st.header("Evaluating the Model's Reliability")
    st.markdown("To trust a model, we must measure its performance on data it has never seen before (the test set).")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
        st.write("This is the percentage of correct predictions the model made on the test set.")
        
        st.write("#### Classification Report")
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.write("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=target_names, yticklabels=target_names, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        st.write("The matrix shows where the model made correct vs. incorrect predictions.")


# --- Feature Importance Tab ---
with tab3:
    st.header("Understanding the 'Why'")
    st.markdown("This chart shows which features were most influential in the model's predictions. Features at the top are the most important.")

    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    feat_imp.plot(kind='barh', ax=ax2, color='skyblue')
    plt.xlabel("Importance Score")
    plt.title("Feature Importances")
    st.pyplot(fig2)