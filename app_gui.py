import customtkinter as ctk
import joblib
from tkinter import messagebox

# =========================
# Load Model and Vectorizer
# =========================
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

try:
    clf = joblib.load(MODEL_PATH)
    vect = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    messagebox.showerror("Error", "Model or vectorizer not found! Train your model first.")
    exit()

# =========================
# Predict Function
# =========================
def predict_sentiment():
    tweet = text_input.get("0.0", "end").strip()
    if not tweet:
        messagebox.showwarning("Input Error", "Please enter a tweet!")
        return

    processed_tweet = vect.transform([tweet])
    prediction = clf.predict(processed_tweet)[0]

    if prediction == 1:
        result_label.configure(text="Positive 😀", text_color="green")
    else:
        result_label.configure(text="Negative 😞", text_color="red")

# =========================
# GUI Setup
# =========================
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"

root = ctk.CTk()
root.title("Twitter Sentiment Analysis")
root.geometry("500x420")
root.resizable(False, False)

# Title
title_label = ctk.CTkLabel(root, text="Twitter Sentiment Analysis", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=15)

# Text box
text_input = ctk.CTkTextbox(root, width=420, height=120, font=ctk.CTkFont(size=14))
text_input.pack(pady=10)

# Predict Button
predict_btn = ctk.CTkButton(root, text="Predict Sentiment", font=ctk.CTkFont(size=14, weight="bold"), command=predict_sentiment)
predict_btn.pack(pady=15)

# Result Label
result_label = ctk.CTkLabel(root, text="", font=ctk.CTkFont(size=18, weight="bold"))
result_label.pack(pady=20)

# Footer
footer_label = ctk.CTkLabel(root, text="Model: TF-IDF + Logistic Regression", font=ctk.CTkFont(size=12), text_color="gray")
footer_label.pack(side="bottom", pady=10)

root.mainloop()
