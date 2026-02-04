import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# 1. Dengelenmiş Veriyi Yükle
df = pd.read_csv(
"data/processed/anket_veri_smote_tfidf.csv")

X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------------------------
# 2. Train / Test Ayrımı

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 3. Naive Bayes Modeli
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# -------------------------------------------------
# 4. Logistic Regression Modeli
lr_model = LogisticRegression(
    max_iter=1000,
    solver="liblinear"
)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)

# -------------------------------------------------
# 5. Sonuçları Yazdır
print("MODEL KARŞILAŞTIRMASI (Accuracy)\n")

print(f"Naive Bayes Accuracy        : {nb_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

#Web için lazım olan modelleri kaydediyorum
import joblib

# Logistic Regression daha iyi olduğu için onu seçtim
joblib.dump(lr_model, "models/skin_model.pkl")


