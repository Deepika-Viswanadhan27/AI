import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset (you can replace with real dataset later)
data = {
    "text": [
        "Government announces new policy",
        "Celebrity caught in fake scandal",
        "Breaking news: market crashes",
        "Fake news spreading on social media"
    ],
    "label": [1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Model
model = LogisticRegression()
model.fit(X, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
