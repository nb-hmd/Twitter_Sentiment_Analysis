## Sentiment_Analysis_on_Twitter_Dataset
Twitter Sentiment Analysis using Sentiment140 dataset with 1.6 million tweets — A Python-based machine learning project that preprocesses 1.6M tweets, trains a TF-IDF + SGDClassifier model, and provides an interactive GUI for real-time sentiment prediction. It demonstrates the full Machine Learning (ML) and Natural Language Processing (NLP) pipeline — from raw data preprocessing to model training, evaluation, and deployment in a user-friendly GUI.

### Working Process
#### 1. Data Collection
Uses the Sentiment140 dataset, which contains tweets labeled as:

0 → Negative

2 → Neutral

4 → Positive

The dataset includes tweet text, IDs, timestamps, and labels.

#### 2. Data Preprocessing (data_preprocessing.py)
Cleans raw tweets by:

Removing URLs, mentions (@username), hashtags (#tag), and special characters.

Converting all text to lowercase for uniformity.

Removing stopwords (common words that don’t contribute to sentiment).

Optional stemming/lemmatization to reduce words to their root form.

Saves the processed dataset as CSV for training.

#### 3. Model Training (tfidf_training.py)
Uses TF-IDF Vectorization to transform text into numerical features.

Trains an SGDClassifier (Stochastic Gradient Descent) for fast, scalable learning.

Performs Hyperparameter Tuning with GridSearchCV to achieve 78% accuracy.

Saves the trained model and TF-IDF vectorizer using joblib in the models/ directory.


#### 4. Graphical User Interface (app_gui.py)
Built with Tkinter for a clean and modern interface.

Features:

Text input box for user tweets or sentences.

"Analyze Sentiment" button to predict sentiment instantly.

Displays results as Positive, Negative, or Neutral.

Attractive design with colors and fonts for better UX.

Loads the saved ML model and vectorizer to make real-time predictions without retraining.

#### 5. Output & Predictions
The GUI returns sentiment instantly based on the trained model.

Handles unseen text efficiently.
