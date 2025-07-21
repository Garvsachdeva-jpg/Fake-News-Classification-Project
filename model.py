import os
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error,r2_score,classification_report,accuracy_score
import joblib
import numpy as np
import matplotlib.pyplot as plt




# product_feedbacks = {
#     "The product is great, I highly recommend it!": "Positive",
#     "This is the worst product I have ever used. It broke after just one day.": "Negative",
#     "It's okay, nothing special.": "Neutral",
#     "I'm very happy with this purchase. It works perfectly.": "Positive",
#     "Terrible quality, a complete waste of money.": "Negative",
#     "I had some issues with the setup, but it works fine now.": "Neutral",
#     "Amazing features and easy to use.": "Positive",
#     "The customer service was unhelpful and rude.": "Negative",
#     "It exceeded my expectations.": "Positive",
#     "Not worth the price, there are better options available.": "Negative",
#     "I like the design, but the performance is not as good as I expected.": "Neutral",
#     "This product has changed my life for the better!": "Positive",
#     "I'm extremely disappointed with this product.": "Negative",
#     "It does what it's supposed to do.": "Neutral",
#     "Highly satisfied with the results.": "Positive",
#     "I regret buying this product.": "Negative",
#     "It's a decent product for the price.": "Neutral",
#     "Fantastic product, a must-have!": "Positive",
#     "Poor build quality, feels cheap.": "Negative",
#     "It's alright, not the best, not the worst.": "Neutral",
#     "I'm impressed with the performance.": "Positive",
#     "This is a scam, don't buy it!": "Negative",
#     "It's a bit tricky to figure out at first.": "Neutral",
#     "So happy with my purchase!": "Positive",
#     "It stopped working after a week.": "Negative",
#     "It's just an average product.": "Neutral",
#     "Incredible value for money.": "Positive",
#     "Very frustrating to use.": "Negative",
#     "It's neither good nor bad.": "Neutral",
#     "Absolutely love this product!": "Positive",
#     "It's a complete rip-off.": "Negative",
#     "It's a standard product, nothing fancy.": "Neutral",
#     "I'm thrilled with the results.": "Positive",
#     "This product is a disaster.": "Negative",
#     "It's okay for occasional use.": "Neutral",
#     "Couldn't be happier with this product.": "Positive",
#     "It's a piece of junk.": "Negative",
#     "It's not great, but it's not terrible either.": "Neutral",
#     "Definitely recommend this product.": "Positive",
#     "It's a waste of time and money.": "Negative",
#     "It's an average performing product.": "Neutral",
#     "Best product I've ever bought.": "Positive",
#     "Worst purchase ever.": "Negative",
#     "It's an acceptable product.": "Neutral",
#     "So impressed with this product.": "Positive",
#     "It's a total failure.": "Negative",
#     "It's an adequate product.": "Neutral",
#     "A game changer!": "Positive",
#     "Don't bother with this product.": "Negative",
#     "It's a middling product.": "Neutral"
# }
# df=pd.DataFrame(list(product_feedbacks.items()),columns=['Feedback','Sentiment'])
# df.to_csv("product_feedbacks.csv",index=False)
df=pd.read_csv(r"C:\Users\LENOVO\OneDrive\Desktop\streamlit ui\combined_file2.csv")
# print(np.where(df["label"]=="REAL"))
# print(df)
# print(df['text'])
stop_words=set(stopwords.words('english'))
print(list(stop_words)[0:10])
lemmatizer=WordNetLemmatizer()
stemmer=PorterStemmer()

def clean_text(text):
  if pd.isna(text):
    return ""
  text=str(text).lower()
  text=re.sub(r"[^a-z\s]","",text)
  tokens=word_tokenize(text)
  cleaned=[lemmatizer.lemmatize(i) for i in tokens if i not in stop_words and len(i)>1]

  return " ".join(cleaned)


# l=clean_text("the product was amazing! But the quality could be better rather than just looks.")
# print(l)
print("cleaning the file......")
df['cleaned']=df['text'].apply(clean_text)


df=df[df['cleaned'].str.len()>0]
print(f"Text cleaning completed  remaining rows after cleaning:{len(df)}")
# s='words'
# s.isalpha()

le=LabelEncoder()
df['Label']=le.fit_transform(df['label'])
print(f"Label encoding done..classes {le.classes_}")

print("converting text to TF-IDF vectors/n")
tfidf=TfidfVectorizer(max_features=5000,min_df=1,max_df=0.95)

X=tfidf.fit_transform(df['cleaned']).toarray()
y=df['Label']

print(f"Tf-idf vectorization completed .Feature matrix shape :{X.shape}")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(random_state=42,max_iter=1000)
model.fit(X_train,y_train)

print("Model training completed!")

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:4f}")
print("Classification Report:")
print(classification_report(y_test,y_pred))



def predict_authenticity(new_text):
    """Predict sentiment of new text"""
    clean = clean_text(new_text)
    if not clean:
        return "neutral", 0.0 

    vector = tfidf.transform([clean])
    prediction = model.predict(vector)
    probability = model.predict_proba(vector)

    sentiment = le.inverse_transform(prediction)[0]
    confidence = max(probability[0])

    return sentiment, confidence

print("\n🧪 Sample Predictions:")
test_texts = [
    "Breaking news on climate change policies in India",
    "Government unveils new education reforms for 2025",
    "Stock markets surge as tech companies report profits",
    "Earthquake of magnitude 6.5 hits the Himalayan region",
    "NASA successfully launches mission to study the sun",
    "Local elections see record voter turnout this year",
    "Health ministry warns against new COVID-19 variant",
    "India signs trade agreement with European Union",
    "Sports update: India defeats Australia in cricket final in 2023",
    "Supreme Court passes judgment on free speech case",
    "Cybersecurity breach affects thousands of users globally"
    ,""
]


for text in test_texts:
    authenticity, confidence = predict_authenticity(text)
    print(f"News: '{text}' → {authenticity} (confidence: {confidence:.3f})")
# Save the model and components
joblib.dump(model, "text_fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(le, "text_label_encoder.pkl")
joblib.dump(accuracy, "model_accuracy.pkl")

# Save classification report to a text file
with open("classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))








