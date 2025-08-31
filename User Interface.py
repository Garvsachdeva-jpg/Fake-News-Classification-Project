import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import nltk
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# === Download NLTK resources ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# === Pattern detection for code ===
def looks_like_code(text):
    # Only flag as code if multiple code-like patterns exist
    patterns = [
        r"\bimport\b", r"\bdef\b", r"\bclass\b", r"[{}]",
        r"<[^>]+>", r"#include\b", r"\bpublic\b", r"\bstatic\b",
        r"\bfunction\b", r"console\.log", r"\bend\b"
    ]
    code_like_matches = sum(1 for p in patterns if re.search(p, text))
    
    # If only a semicolon is found, but it's natural language, ignore
    semicolon_count = text.count(";")
    
    # Require at least 2 code-like matches or 1 strong pattern (like `import`) AND no full sentences
    return code_like_matches >= 2 or (semicolon_count >= 2 and code_like_matches >= 1)

# === Load Models ===
try:
    model = joblib.load("text_fake_news_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("text_label_encoder.pkl")
    classification_report_text = open("classification_report.txt").read()
    model_accuracy = round(float(joblib.load("model_accuracy.pkl")), 4)
except Exception as e:
    st.error(f"⚠️ Model loading failed: {e}")
    st.stop()

# === Preprocessing ===
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
#cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(cleaned)

# === Prediction ===
def predict_authenticity(text):
    cleaned = clean_text(text)
    if not cleaned:
        return "neutral", 0.0, cleaned
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)
    proba = model.predict_proba(vector)
    label = le.inverse_transform(pred)[0]
    confidence = float(np.max(proba))
    return label, confidence, cleaned

# === Logging ===
LOG_FILE = "classification_logs.csv"

# Create log file if not exists
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "input_text", "cleaned_text", "prediction", "confidence"]).to_csv(LOG_FILE, index=False)

def log_prediction(input_text, cleaned_text, prediction, confidence):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": input_text,
        "cleaned_text": cleaned_text,
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }
    log_df = pd.DataFrame([log_entry])
    log_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# === Streamlit UI ===
st.set_page_config(page_title="📰 Fake News Classifier", layout="centered")
st.title("📰 Fake News Classifier")
st.write("🔍 Check if a news headline is **REAL** or **FAKE** with our smart classifier.")

text_input = st.text_area("✍️ Enter your news headline or short paragraph here:", height=150)

if st.button("🚨 Check Now"):
    if text_input.strip():
        with st.spinner("Analyzing text..."):
            if looks_like_code(text_input):
                st.warning("⚠️ This seems like code. Try pasting actual news content.")
            elif len(text_input.strip()) < 5:
                st.warning("⚠️ The input is too short to evaluate. Please enter a longer news snippet.")
            else:
                result, confidence, cleaned = predict_authenticity(text_input)
                log_prediction(text_input, cleaned, result, confidence)

                # Display result based on confidence bands
                if confidence >= 0.8:
                    st.success(f"🟢 **Prediction:** `{result.upper()}`")
                    st.caption("✅ This prediction has **high confidence**.")
                elif 0.6 <= confidence < 0.8:
                    st.info(f"🟡 **Prediction:** `{result.upper()}`")
                    st.caption("⚠️ This is a **moderate confidence** prediction. Please verify from other sources.")
                else:
                    st.warning("🟠 **Prediction:** `UNCERTAIN`")
                    st.caption("❌ This result has **low confidence**. Try improving or lengthening the input.")

                # Display progress and confidence
                st.progress(confidence)
                st.info(f"📊 **Confidence Score:** `{confidence:.2f}`")
                st.caption("Confidence reflects how sure the model is about this prediction based on your input.")

                # Optional suggestions if confidence is low
                if confidence < 0.6:
                    st.markdown("🔍 **Tips to Improve Prediction:**")
                    st.markdown("- Use a complete news sentence or paragraph.")
                    st.markdown("- Avoid informal or mixed-language text.")
                    st.markdown("- Include names, dates, or specific references.")
    else:        
        st.warning("⚠️ Please enter a news snippet for analysis.")

# === Sidebar: Interactive Model Info ===
st.sidebar.title("🧠 AI Model Info")

with st.sidebar.expander("🔍 How It Works"):
    st.markdown("- Uses **TF-IDF vectorizer** for feature extraction.\n"
                "- Classified using **Logistic Regression**.\n"
                "- Trained on 10K+ labeled news headlines.")

st.sidebar.metric("📈 Accuracy", f"{model_accuracy * 100:.2f}%")
st.sidebar.markdown("✅ **Vectorizer:** TF-IDF\n✅ **Labels:** Fake, Real")

with st.sidebar.expander("📊 Classification Report"):
    st.code(classification_report_text, language="text")

with st.sidebar.expander("📁 View Last 5 Logs"):
    try:
        logs_df = pd.read_csv(LOG_FILE).tail(5)
        st.dataframe(logs_df)
    except Exception as e:
        st.warning(f"Unable to load logs: {e}")

with st.sidebar.expander("🚀 Future Scope"):
    st.markdown("""
    - 🔗 Embed in news platforms (e.g., `NDTV`, `India Today`) to auto-flag suspicious headlines.
    - 🧑‍💻 Combine with **browser extensions** for real-time detection.
    - 📱 Mobile app version using **React Native**.
    - 📈 Admin dashboard to view flagged news in bulk.
    """)

# === Footer ===
st.markdown("---")
st.markdown("🛠️ Built with ❤️ using **scikit-learn**, **NLTK**, and **Streamlit**")
st.markdown("📅 2025 | ✨ Designed to be extended into full-scale **news analysis platforms**.")
