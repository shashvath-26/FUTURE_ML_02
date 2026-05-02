import pandas as pd
import numpy as np
import re
import nltk
import joblib
import warnings

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# Download NLTK resources if not already available
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


class TextPreprocessor:
    """Simple text cleaning and preprocessing for support tickets."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = " ".join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)

    def preprocess(self, texts):
        processed_texts = []

        for text in texts:
            cleaned_text = self.clean_text(text)
            final_text = self.tokenize_and_lemmatize(cleaned_text)
            processed_texts.append(final_text)

        return processed_texts


class TicketClassifier:
    """Support ticket classification model."""

    def __init__(self, model_type="naive_bayes"):
        self.model_type = model_type
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.model = self.get_model()
        self.categories = None

    def get_model(self):
        if self.model_type == "naive_bayes":
            return MultinomialNB()
        elif self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            return MultinomialNB()

    def train(self, df, text_col="text", category_col="category"):
        print("Preprocessing text data...")
        processed_texts = self.preprocessor.preprocess(df[text_col].fillna(""))

        X = self.vectorizer.fit_transform(processed_texts)
        y = df[category_col]

        self.categories = sorted(y.unique())

        # Only use stratified split if all classes have at least 2 samples
        stratify_param = y if y.value_counts().min() >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=stratify_param
        )

        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 50)
        print(f"Model Evaluation ({self.model_type})")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return accuracy

    def predict_category(self, text):
        processed_text = self.preprocessor.preprocess([text])[0]
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_text)[0]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized_text)[0]
            confidence = float(np.max(probabilities))
            all_probabilities = dict(zip(self.model.classes_, probabilities))
        else:
            confidence = None
            all_probabilities = None

        return {
            "category": prediction,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }

    def predict_priority(self, text):
        text_lower = text.lower()

        high_priority_words = [
            "urgent", "immediately", "emergency", "critical",
            "cannot access", "account hacked", "security",
            "charged twice", "refund", "down", "crash"
        ]

        medium_priority_words = [
            "slow", "not working", "error", "bug",
            "help", "guide", "how to", "question"
        ]

        for word in high_priority_words:
            if word in text_lower:
                return "high"

        for word in medium_priority_words:
            if word in text_lower:
                return "medium"

        return "low"

    def classify_ticket(self, text):
        category_result = self.predict_category(text)
        priority = self.predict_priority(text)

        return {
            "text": text,
            "category": category_result["category"],
            "confidence": category_result["confidence"],
            "priority": priority
        }

    def save_model(self, filepath="ticket_classifier_model.pkl"):
        model_data = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "preprocessor": self.preprocessor,
            "categories": self.categories
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="ticket_classifier_model.pkl"):
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.vectorizer = model_data["vectorizer"]
        self.preprocessor = model_data["preprocessor"]
        self.categories = model_data["categories"]
        print(f"Model loaded from {filepath}")


def main():
    print("=" * 60)
    print("SUPPORT TICKET CLASSIFICATION SYSTEM")
    print("FUTURE Internship - Task 2")
    print("=" * 60)

    print("\nLoading dataset...")
    df = pd.read_csv("support_tickets_dataset.csv")

    print(f"Dataset loaded successfully: {len(df)} tickets")
    print("\nDataset columns:")
    print(list(df.columns))

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nPriority distribution:")
    print(df["priority"].value_counts())

    print("\n" + "=" * 60)
    classifier = TicketClassifier(model_type="naive_bayes")
    accuracy = classifier.train(df)

    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE TICKETS")
    print("=" * 60)

    test_tickets = [
        "I cannot login to my account! This is very urgent!",
        "The app is loading very slowly, can you help?",
        "Great service! I love your platform.",
        "My payment was declined but money was deducted.",
        "Can you add a dark mode feature?"
    ]

    for ticket in test_tickets:
        result = classifier.classify_ticket(ticket)
        confidence = result["confidence"]

        print(f"\nTicket: {ticket}")
        print(f"Category: {result['category']}")
        if confidence is not None:
            print(f"Confidence: {confidence:.2%}")
        else:
            print("Confidence: N/A")
        print(f"Priority: {result['priority']}")

    classifier.save_model()

    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 60)

    return classifier


if __name__ == "__main__":
    classifier = main()
