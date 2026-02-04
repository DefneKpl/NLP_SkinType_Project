from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# -------------------------------------------------
# Flask app
app = Flask(__name__)

# -------------------------------------------------
# Model ve TF-IDF yükle
model = joblib.load("models/skin_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# -------------------------------------------------
# NLTK ayarları
nltk.download("stopwords")
stop_words = set(stopwords.words("turkish"))

# -------------------------------------------------
# Ön işleme 
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# -------------------------------------------------
# Ana sayfa
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        user_text = request.form["text"]

        clean_text = preprocess_text(user_text)
        vector = tfidf.transform([clean_text])
        prediction = model.predict(vector)[0]

    return render_template("index.html", prediction=prediction)

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
