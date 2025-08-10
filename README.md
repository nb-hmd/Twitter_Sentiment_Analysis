#  Twitter Sentiment Analysis (Sentiment140 Dataset)

This project performs **Sentiment Analysis** on Twitter data using the [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/datasets/kazanova/sentiment140) containing **1.6 million tweets**.  
It classifies tweets as **Positive 😀** or **Negative 😞** using a **TF-IDF + Logistic Regression** model with hyperparameter tuning to achieve **85%+ accuracy**.

Additionally, a **modern GUI** (built with `CustomTkinter`) allows real-time prediction of tweet sentiment.


##  Project Structure
📂 Twitter Sentiment Analysis
│
├── sentiment140.csv # Raw dataset (1.6M tweets)
│
├── data_preprocessing.py # Cleans raw CSV data, saves preprocessed CSV
├── 📂 data
│    │
│    └── cleaned_sentiment140.csv # Preprocessed dataset
│
├── tfidf_training.py # Trains TF-IDF + Logistic Regression model
├── app_gui.py # Modern GUI for real-time sentiment prediction
│
├── 📂 models
│     │
│     ├── sentiment_model.pkl # Saved trained model
│     └── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
│
├── README.md # Project documentation
└── requirements.txt # Dependencies list


## Models Used

#### TF-IDF Vectorizer
Converts tweets into numerical features, giving more weight to important terms and reducing the impact of common words.

#### Logistic Regression (with Hyperparameter Tuning)
Selected for its speed, interpretability, and strong performance in binary classification problems.
Tuned using:
     - C (regularization strength)
     - max_iter
     - class_weight
     - solver


## 🔄 Workflow
#### Data Preprocessing (data_preprocessing.py)
- Loads the Sentiment140 dataset
- Renames columns: polarity, id, date, query, user, text
- Cleans tweets: lowercasing, removing mentions, URLs, special chars
- Maps polarity:
    - 0 → Negative
    - 4 → Positive
- Saves cleaned dataset to cleaned_sentiment140.csv

#### Model Training (tfidf_training.py)
- Splits dataset into train/test sets
- Converts text into TF-IDF features
- Tunes Logistic Regression hyperparameters
- Achieves 78% accuracy
- Saves sentiment_model.pkl and tfidf_vectorizer.pkl

#### GUI Prediction (app_gui.py)

- Built with CustomTkinter for modern UI (dark mode, rounded buttons)
- Takes tweet as input
- Displays sentiment in real-time with emoji feedback


## ⚙ Dependencies

Install the required dependencies with:

```bash
pip install -r requirements.txt
