# FUTURE_ML_02

## Support Ticket Classification using NLP and Scikit-learn

**FUTURE Internship - Task 2 (Machine Learning)**

---

## Project Overview

This project implements an automated support ticket classification system that:
- **Cleans and preprocesses** text data using NLP techniques
- **Classifies tickets** into different categories (bug, billing, security, etc.)
- **Assigns priority levels** (high/medium/low) based on ticket content
- **Evaluates model performance** using standard ML metrics

---

## Features

- **Text Preprocessing**: Lowercase conversion, URL/email removal, tokenization, stopword removal, and lemmatization
- **Multiple ML Models**: Naive Bayes, Logistic Regression, and Random Forest classifiers
- **TF-IDF Vectorization**: Converts text into numerical features for ML models
- **Priority Tagging**: Keyword-based priority assignment for tickets
- **Model Persistence**: Save and load trained models using joblib
- **Comprehensive Evaluation**: Accuracy, classification report, and confusion matrix

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shashvath-26/FUTURE_ML_02.git
cd FUTURE_ML_02
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (handled automatically by the script):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Usage

### Running the Python Script

```bash
python support_ticket_classifier.py
```

This will:
- Load the support tickets dataset
- Train the classifier model
- Display evaluation metrics
- Test with sample tickets
- Save the trained model

### Using the Classifier in Your Code

```python
from support_ticket_classifier import TicketClassifier
import pandas as pd

# Initialize classifier
classifier = TicketClassifier(model_type='naive_bayes')

# Load and train
df = pd.read_csv('support_tickets_dataset.csv')
classifier.train(df, text_col='text', category_col='category')

# Classify a new ticket
result = classifier.classify_ticket("I cannot login to my account!")
print(f"Category: {result['category']}")
print(f"Priority: {result['priority']}")
print(f"Confidence: {result['confidence']:.2%}")

# Save the model
classifier.save_model('my_model.pkl')

# Load a saved model
classifier.load_model('my_model.pkl')
```

---

## Project Structure

```
FUTURE_ML_02/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file (Python)
├── support_tickets_dataset.csv    # Training dataset (50 tickets)
├── support_ticket_classifier.py   # Main classifier implementation
└── ticket_classifier_model.pkl    # Saved trained model (generated)
```

---

## Dataset

The dataset (`support_tickets_dataset.csv`) contains 50 labeled support tickets with the following columns:
- **ticket_id**: Unique identifier for each ticket
- **text**: The support ticket content
- **category**: Ticket category (bug, billing, security, feature_request, etc.)
- **priority**: Priority level (high, medium, low)

### Categories Included:
- bug
- billing
- security
- login_issue
- account
- feature_request
- performance
- feedback
- integration
- notification
- complaint
- support
- outage
- data_recovery
- navigation
- legal
- documentation

---

## Model Evaluation

The classifier uses an 80/20 train-test split with stratification to ensure balanced class distribution. Evaluation metrics include:
- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, Recall, F1-Score per class
- **Confusion Matrix**: Visual representation of predictions vs actual labels

---

## Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Machine Learning models
- **NLTK**: Natural Language Processing
- **spaCy**: Advanced NLP (optional)
- **Joblib**: Model persistence
- **Matplotlib/Seaborn**: Visualization (optional)

---

## How It Works

1. **Data Loading**: Load support tickets from CSV
2. **Text Preprocessing**: Clean and normalize text
3. **Feature Extraction**: Convert text to TF-IDF vectors
4. **Model Training**: Train ML classifier on labeled data
5. **Evaluation**: Assess model performance on test set
6. **Prediction**: Classify new tickets with category and priority

---

## Future Improvements

- Add deep learning models (LSTM, BERT)
- Implement multi-label classification
- Add web interface for ticket submission
- Include model comparison dashboard
- Add data augmentation for better performance
- Implement active learning for continuous improvement

---

## Author

**shashvath-26**

Project created as part of the FUTURE Internship - Machine Learning Task 2.

---

## License

This project is for educational purposes as part of the FUTURE Internship program.
