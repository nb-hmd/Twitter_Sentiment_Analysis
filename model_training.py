import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import time

# Paths
PROCESSED_DATA_PATH = "data/cleaned_tweets.csv"
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def train_model():
    print("Loading cleaned dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Drop empty rows
    df.dropna(subset=["text", "polarity"], inplace=True)

    X = df["text"]
    y = df["polarity"]

    print(f"Dataset size: {len(X)} rows")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=5000,  # adjust for speed/accuracy
        ngram_range=(1, 2)  # unigrams + bigrams
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Logistic Regression with Grid Search
    print("Starting hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [200, 300],
        'solver': ['liblinear', 'lbfgs']
    }

    grid = GridSearchCV(
        LogisticRegression(),
        param_grid,
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    start_time = time.time()
    grid.fit(X_train_tfidf, y_train)
    elapsed_time = time.time() - start_time
    print(f"Training & tuning took {elapsed_time:.2f} seconds")

    best_model = grid.best_estimator_
    print(f"Best parameters: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model and vectorizer
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_model()
