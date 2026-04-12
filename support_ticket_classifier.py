"""
Support Ticket Classification System
FUTURE Internship - Task 2 (Machine Learning)

This module provides a complete support ticket classification system that:
- Cleans and preprocesses text data
- Classifies tickets into categories
- Assigns priority levels (high/medium/low)
- Evaluates model performance

Author: shashvath-26
Project: FUTURE_ML_02
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Handles text cleaning and preprocessing for support tickets."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """Clean and preprocess a single text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text."""
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    
    def preprocess(self, texts):
        """Preprocess a list of texts."""
        cleaned = [self.clean_text(text) for text in texts]
        processed = [self.tokenize_and_lemmatize(text) for text in cleaned]
        return processed


class TicketClassifier:
    """Main classifier for support ticket categorization."""
    
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = MultinomialNB()
        
        self.preprocessor = TextPreprocessor()
        self.categories = None
        self.priorities = None
    
    def train(self, df, text_col='text', category_col='category'):
        """Train the classifier on the provided dataset."""
        print("Preprocessing text data...")
        processed_texts = self.preprocessor.preprocess(df[text_col])
        
        self.categories = df[category_col].unique()
        print(f"Categories found: {list(self.categories)}")
        
        X = self.vectorizer.fit_transform(processed_texts)
        y = df[category_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Model Evaluation ({self.model_type})")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict_category(self, text):
        """Predict the category of a single ticket."""
        processed = self.preprocessor.preprocess([text])[0]
        vectorized = self.vectorizer.transform([processed])
        prediction = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        
        return {
            'category': prediction,
            'confidence': max(probabilities),
            'all_probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def predict_priority(self, text):
        """Assign priority level based on keywords in the text."""
        text_lower = text.lower()
        
        high_priority_keywords = [
            'urgent', 'immediately', 'emergency', 'critical',
            'cannot access', 'account hacked', 'security',
            'charged twice', 'refund', 'down', 'crash'
        ]
        
        medium_priority_keywords = [
            'slow', 'not working', 'error', 'bug',
            'help', 'guide', 'how to', 'question'
        ]
        
        for keyword in high_priority_keywords:
            if keyword in text_lower:
                return 'high'
        
        for keyword in medium_priority_keywords:
            if keyword in text_lower:
                return 'medium'
        
        return 'low'
    
    def classify_ticket(self, text):
        """Full classification of a ticket with category and priority."""
        category_result = self.predict_category(text)
        priority = self.predict_priority(text)
        
        return {
            'text': text,
            'category': category_result['category'],
            'confidence': category_result['confidence'],
            'priority': priority
        }
    
    def save_model(self, filepath='ticket_classifier_model.pkl'):
        """Save the trained model and vectorizer."""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'categories': self.categories
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ticket_classifier_model.pkl'):
        """Load a trained model and vectorizer."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.categories = model_data['categories']
        print(f"Model loaded from {filepath}")


def main():
    """Main function to demonstrate the classifier."""
    print("="*60)
    print("SUPPORT TICKET CLASSIFICATION SYSTEM")
    print("FUTURE Internship - Task 2")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('support_tickets_dataset.csv')
    print(f"Dataset loaded: {len(df)} tickets")
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())
    print(f"\nPriority distribution:")
    print(df['priority'].value_counts())
    
    # Train classifier
    print("\n" + "="*60)
    classifier = TicketClassifier(model_type='naive_bayes')
    accuracy = classifier.train(df)
    
    # Test with sample tickets
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE TICKETS")
    print("="*60)
    
    test_tickets = [
        "I cannot login to my account! This is very urgent!",
        "The app is loading very slowly, can you help?",
        "Great service! I love your platform.",
        "My payment was declined but money was deducted.",
        "Can you add a dark mode feature?"
    ]
    
    for ticket in test_tickets:
        result = classifier.classify_ticket(ticket)
        print(f"\nTicket: {ticket}")
        print(f"  Category: {result['category']} (Confidence: {result['confidence']:.2%})")
        print(f"  Priority: {result['priority']}")
    
    # Save the model
    classifier.save_model()
    
    print("\n" + "="*60)
    print("CLASSIFICATION COMPLETE!")
    print("="*60)
    
    return classifier


if __name__ == '__main__':
    classifier = main()
