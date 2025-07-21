# 📰 Fake News Classifier 🚨

A smart web app to detect **fake vs real news headlines** using NLP and machine learning (TF-IDF + Logistic Regression). Built with **Streamlit** for an interactive and responsive user experience.

---

## 🔍 Features

- ✅ Detect if a news headline is **FAKE** or **REAL**
- 🔠 Text cleaning: tokenization, lemmatization, stopword removal
- 📊 Confidence-based prediction with visual feedback
- 📈 Trained on 10K+ labeled news headlines
- 🧠 ML pipeline: **TF-IDF Vectorizer + Logistic Regression**
- 🗂️ Logs predictions with timestamp into a CSV file
- 💡 Suggests how to improve input if confidence is low
- 📁 View the **last 5 predictions** from the sidebar

---

## ⚙️ Technologies Used

| Component        | Tool/Library         |
|------------------|----------------------|
| UI & Frontend    | [Streamlit](https://streamlit.io/)         |
| NLP              | [NLTK](https://www.nltk.org/)              |
| ML Model         | [Scikit-learn](https://scikit-learn.org/)  |
| Persistence      | Pandas, CSV (Logging) |
| Model Format     | Joblib `.pkl` files   |

---

## 🏗️ Project Structure


├── ui_app.py # Streamlit UI & logic
├── text_fake_news_model.pkl # Trained Logistic Regression model
├── tfidf_vectorizer.pkl # TF-IDF transformer
├── text_label_encoder.pkl # Label encoder
├── model_accuracy.pkl # Saved model accuracy
├── classification_report.txt # Precision/recall/F1 metrics
├── classification_logs.csv # Logs of all predictions (auto-generated)
