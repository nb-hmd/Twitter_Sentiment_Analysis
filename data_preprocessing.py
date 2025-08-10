import pandas as pd
import re
import string
import emoji
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

RAW_DATA_PATH = "Sentiment140 dataset with 1.6 million tweets.csv"  # path to your raw dataset
PROCESSED_DATA_PATH = "data/cleaned_tweets.csv"
CHUNK_SIZE = 200_000  # adjust based on your RAM

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans a tweet's text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r"@\w+", '', text)  # remove mentions
    text = re.sub(r"#\w+", '', text)  # remove hashtags
    text = emoji.replace_emoji(text, replace='')  # remove emojis
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def process_and_save():
    print("Starting preprocessing...")
    chunk_list = []
    start_row = 0
    chunk_number = 1

    while True:
        print(f"Processing chunk {chunk_number} (rows {start_row + 1} - {start_row + CHUNK_SIZE})...")
        try:
            chunk = pd.read_csv(
                RAW_DATA_PATH,
                encoding='latin-1',
                header=None,
                names=["polarity", "id", "date", "query", "user", "text"],
                skiprows=start_row,
                nrows=CHUNK_SIZE
            )
        except pd.errors.EmptyDataError:
            break

        if chunk.empty:
            break

        # Keep only needed columns
        chunk = chunk[["polarity", "text"]]

        # Convert polarity to binary (0=negative, 1=positive)
        chunk["polarity"] = chunk["polarity"].replace(4, 1)

        # Clean tweets
        chunk["text"] = chunk["text"].astype(str).apply(clean_text)

        chunk_list.append(chunk)

        start_row += CHUNK_SIZE
        chunk_number += 1

    cleaned_df = pd.concat(chunk_list, ignore_index=True)
    print(f"Saving cleaned data to {PROCESSED_DATA_PATH}...")
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    process_and_save()
