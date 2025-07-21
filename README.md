# Fake-News-Classification-Project
| Component     | Tool/Library                              |
| ------------- | ----------------------------------------- |
| UI & Frontend | [Streamlit](https://streamlit.io/)        |
| NLP           | [NLTK](https://www.nltk.org/)             |
| ML Model      | [Scikit-learn](https://scikit-learn.org/) |
| Persistence   | Pandas, CSV (Logging)                     |
| Model Format  | Joblib (.pkl files)                       |

#Project Structure
├── ui_app.py                   # Streamlit UI & logic
├── text_fake_news_model.pkl    # Trained Logistic Regression model
├── tfidf_vectorizer.pkl        # TF-IDF transformer
├── text_label_encoder.pkl      # Label encoder
├── model_accuracy.pkl          # Saved model accuracy
├── classification_report.txt   # Precision/recall/F1 metrics
├── classification_logs.csv     # Logs of all predictions (auto-generated)
📌 Future Improvements
🌐 Browser extension for real-time detection

📱 React Native mobile version

🧑‍💻 Admin dashboard to bulk review flagged headlines

☁️ Migrate logging to a cloud database (e.g., Firebase or MongoDB)
