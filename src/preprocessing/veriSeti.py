import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# -------------------------------------------------
# 1. NLTK(natural language toolkit) ayarları

nltk.download("stopwords")
stop_words = set(stopwords.words("turkish"))

# -------------------------------------------------
# 2. Veriyi Yükle (ANKET)

df = pd.read_csv("data/raw/Skin_text_dataset.csv")
# kolonlar: text | label
print("Orijinal sınıf dağılımı:")
print(df["label"].value_counts()) #Normal:11, Kuru:6, Yağlı:11, Karma:11

# -------------------------------------------------
# 3. Ön İşleme

def preprocess_text(text):
    # Küçük harf
    text = text.lower()
    # Noktalama ve sayıları kaldır
    text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    # Tokenization
    tokens = text.split()
    # Stopword + kısa kelime temizliği
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess_text)

# -------------------------------------------------
# 4. TF-IDF (Sayısallaştırma)

tfidf = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2
)

X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

# -------------------------------------------------
# 5. SMOTE ile Dengeleme 

smote = SMOTE(
    sampling_strategy={
        "Kuru": 30,
        "Yağlı": 30,
        "Karma": 30,
        "Normal": 30
    },
    random_state=42
)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nSMOTE sonrası sınıf dağılımı:")
print(pd.Series(y_resampled).value_counts())

# -------------------------------------------------
# 6. Dengelenmiş Veriyi DataFrame'e Çevirme

X_resampled_df = pd.DataFrame(
    X_resampled.toarray(),
    columns=tfidf.get_feature_names_out()
)

balanced_df = X_resampled_df.copy()
balanced_df["label"] = y_resampled

# -------------------------------------------------
# 7. Kaydet

balanced_df.to_csv(
    "data/processed/anket_veri_smote_tfidf.csv",
    index=False,
    encoding="utf-8"
)

print("\nSMOTE ile dengelenmiş veri kaydedildi.")


# TF-IDF’i kaydediyorum web kısmı için lazım
import joblib
joblib.dump(tfidf, "models/tfidf.pkl")

