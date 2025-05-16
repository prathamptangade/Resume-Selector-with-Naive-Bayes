import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import CountVectorizer

def load_and_process_data(file):
    # Load Excel
    df = pd.read_excel(file, engine='openpyxl')
    
    # Clean up
    df = df[df['resume_text'] != "False"]  # remove invalid rows
    df = df.dropna(subset=['resume_text', 'class'])  # drop NaNs
    df['resume_text'] = df['resume_text'].astype(str)
    
    # Features and labels
    X_raw = df['resume_text']
    y = df['class']  # or df['label'] if renamed
    
    # Convert text to numeric features
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X_raw)
    
    return X, y


def train_model(X, y, test_size=0.3):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train Naive Bayes model
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    
    # Predict
    y_pred = nb_classifier.predict(X_test)
    
    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return cm, report, y_pred, y_test

def plot_confusion_matrix(cm):
    # Plot confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Convert plot to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str
