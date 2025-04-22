"""Streamlit Twitter Sentiment Analysis App with Auth, Models, and Evaluation."""

import os
import re
import pickle
import sqlite3
import nltk
from nltk.corpus import stopwords
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ntscraper import Nitter

# ---------------------- DB CONNECTION ----------------------
CONN = sqlite3.connect('user_credentials.db', check_same_thread=False)
CURSOR = CONN.cursor()


def create_users_table():
    """Create user table if not exists."""
    CURSOR.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        name TEXT,
        password TEXT,
        email TEXT
    )
    ''')
    CONN.commit()


def register_user(username, name, password, email):
    """Insert user into DB."""
    try:
        CURSOR.execute("INSERT INTO users (username, name, password, email) VALUES (?, ?, ?, ?)",
                       (username, name, password, email))
        CONN.commit()
        return True, "Registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def login_user(username, password):
    """Authenticate user login."""
    CURSOR.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    return CURSOR.fetchone()


# ---------------------- PREPROCESSING ----------------------
@st.cache_resource
def load_stopwords():
    """Load NLTK English stopwords."""
    nltk.download('stopwords')
    return stopwords.words('english')


def preprocess_text(text, stop_words):
    """Clean and preprocess input text."""
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)


# ---------------------- LOAD MODELS ----------------------
@st.cache_resource
def load_models():
    """Load all models except vectorizer."""
    models = {}
    for file in os.listdir("models"):
        if file.endswith(".pkl") and "vectorizer" not in file:
            model_name = file.replace(".pkl", "")
            with open(f"models/{file}", "rb") as file_obj:
                models[model_name] = pickle.load(file_obj)
    return models


@st.cache_resource
def load_vectorizer():
    """Load vectorizer from disk."""
    with open("models/vectorizer.pkl", "rb") as file_obj:
        return pickle.load(file_obj)


# ---------------------- PREDICT ----------------------
def predict_sentiment(text, model, vectorizer, stop_words):
    """Predict sentiment using model and vectorizer."""
    clean = preprocess_text(text, stop_words)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)
    return "Positive" if prediction[0] == 1 else "Negative"


# ---------------------- SCRAPER ----------------------
@st.cache_resource
def initialize_scraper():
    """Initialize Nitter scraper."""
    return Nitter(log_level=1)


# ---------------------- STREAMLIT SECTIONS ----------------------
def handle_manual_input(model, vectorizer, stop_words):
    """Handle manual text sentiment input."""
    input_text = st.text_area("Enter your text here:")
    if st.button("Analyze Sentiment"):
        result = predict_sentiment(input_text, model, vectorizer, stop_words)
        st.success(f"Predicted Sentiment: {result}")


def handle_tweet_input(scraper, model, vectorizer, stop_words):
    """Handle fetching tweets and analyzing sentiment."""
    handle = st.text_input("Twitter Username (public account)")
    if st.button("Fetch Tweets"):
        tweets = scraper.get_tweets(handle, mode='user', number=5)
        if 'tweets' in tweets:
            for tweet in tweets['tweets']:
                text = tweet['text']
                sentiment = predict_sentiment(text, model, vectorizer, stop_words)
                st.markdown(f"**{sentiment}**: {text}")
        else:
            st.error("No tweets found or account is private.")


def show_evaluation_button(model, vectorizer, stop_words):
    """Show evaluation scores."""
    st.markdown("### Model Evaluation (Example)")
    if st.button("Show Example Evaluation"):
        try:
            with open("models/sample_data.pkl", "rb") as file_obj:
                x_test, y_test = pickle.load(file_obj)
            x_vec = vectorizer.transform([preprocess_text(x, stop_words) for x in x_test])
            y_pred = model.predict(x_vec)
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
            st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
            st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
            st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
        except (FileNotFoundError, pickle.PickleError) as err:
            st.error(f"Error loading evaluation data: {err}")


def login_flow():
    """Login logic and app functionality."""
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_data = login_user(user, password)
        if user_data:
            st.success(f"Welcome {user_data[1]} 👋")

            stop_words = load_stopwords()
            models = load_models()
            vectorizer = load_vectorizer()
            scraper = initialize_scraper()

            st.subheader("Choose a Model for Sentiment Analysis")
            selected_model = st.selectbox("Select a model", list(models.keys()))
            model = models[selected_model]

            st.write("### Test the model:")
            option = st.radio("Choose Input Type", ["Manual Text", "Get Tweets from Username"])
            if option == "Manual Text":
                handle_manual_input(model, vectorizer, stop_words)
            else:
                handle_tweet_input(scraper, model, vectorizer, stop_words)

            show_evaluation_button(model, vectorizer, stop_words)
        else:
            st.error("Invalid username or password")


# ---------------------- STREAMLIT APP ----------------------
def main():
    """Streamlit app main function."""
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    st.title("Twitter Sentiment Analysis App")

    create_users_table()
    auth_mode = st.sidebar.radio("Choose Auth Mode", ["Register", "Login"])

    if auth_mode == "Register":
        st.subheader("Register")
        name = st.text_input("Name")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            success, msg = register_user(username, name, password, email)
            if success:
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.subheader("Login")
        login_flow()


if __name__ == "__main__":
    main()