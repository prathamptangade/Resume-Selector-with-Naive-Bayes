# Resume Selector – NLP Classification with Naive Bayes

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31014/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Resume Classification System** using **Natural Language Processing (NLP)** and **Multinomial Naive Bayes** to categorize resumes as **flagged** or **not flagged**. This project includes a **Jupyter Notebook** for model training and a **Streamlit Web App** for interactive resume classification with Excel dataset uploads.

## 🚀 Features
- **Data Processing**: Load and clean resume text from Excel (`.xlsx`) files.
- **NLP Pipeline**: Convert resume text to numerical features using **CountVectorizer** (with planned TF-IDF support).
- **Model Training**: Train a **Multinomial Naive Bayes** classifier for binary classification.
- **Visualization**: Display **confusion matrix** heatmap and **classification report** (precision, recall, F1-score).
- **Interactive App**: Streamlit interface to upload datasets, adjust test split, and view results in real-time.
- **Error Handling**: Robust handling of missing data and invalid entries.

## 📁 File Structure


├── app.py                   # Streamlit app interface

├── resume_selector.py       # Core logic: data loading, model training, plotting

├── resume_selector.ipynb    # Jupyter Notebook for model training and analysis

├── sample_dataset.xlsx      # Sample Excel dataset





## 📦 Installation
1. **Clone the repository** (if using Git, optional):
   ```bash
   git clone <your-repo-url>
   cd resume-selector
Create a virtual environment:
bash

Copy
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
Install dependencies:
bash

Copy
pip install pandas numpy scikit-learn matplotlib seaborn nltk openpyxl streamlit
📊 Input Dataset Requirements
Upload an Excel file (.xlsx) with at least two columns:

resume_text: Text content of each resume (e.g., "Experienced in Python…").
class: Binary label (flagged or not flagged).
Sample Dataset (sample_dataset.xlsx):


resume_text	class
Experienced in Python…	flagged
Project management…	not flagged
Note: Ensure text data is clean (no invalid characters). Missing values are handled automatically.

🧪 Using the Jupyter Notebook
Open resume_selector.ipynb in Jupyter Notebook or JupyterLab.
Follow the steps to:
Load and preprocess the Excel dataset.
Vectorize resume_text using CountVectorizer.
Train the Naive Bayes model.
Evaluate with:
Confusion Matrix heatmap.
Classification Report (precision, recall, F1-score).
Accuracy score.


🌐 Running the Streamlit Web App
Activate the virtual environment:
bash

Copy
.\venv\Scripts\activate  # Windows
Run the app:
bash


Copy
streamlit run app.py
Open http://localhost:8501 in your browser.

App Functionality
📂 Upload: Select an .xlsx file with resume_text and class columns.
🔧 Test Split: Adjust test set size (0.1 to 0.5) via a slider.
⚙️ Run Model:
Trains the Naive Bayes classifier.
Displays confusion matrix heatmap.
Shows classification report and predictions (actual vs. predicted).
Reports accuracy percentage.


📈 Preview: View the first 5 rows of the dataset.
🛠 Code Overview
resume_selector.py
load_and_process_data(file):
Loads Excel file, cleans data, and vectorizes resume_text using CountVectorizer.
Returns features (X) and labels (y).
train_model(X, y, test_size):
Splits data, trains Naive Bayes, and generates metrics.
plot_confusion_matrix(cm):
Creates a confusion matrix heatmap as a base64-encoded PNG.
app.py
Streamlit interface for file upload, model training, and result visualization.
Handles errors gracefully with user-friendly messages.


📈 Example Output (Web App)

Section	Description
Dataset Preview	Table showing first 5 rows of uploaded data.
Confusion Matrix	Heatmap visualizing classification results.
Classification Report	Precision, recall, F1-score per class.
Predictions	Table of actual vs. predicted labels.
Accuracy	Overall model accuracy (e.g., 87%).
 (Add screenshot for visual appeal)

🔮 Future Enhancements
Implement TF-IDF or transformer-based text encodings (e.g., BERT).
Add visualization for misclassified resumes.
Enable export of predictions to CSV/Excel.
Support multi-class classification (e.g., multiple job roles).
Deploy to Streamlit Community Cloud for online access.

👨‍💻 Author
Pratham Tangade

Computer Science student passionate about NLP and machine learning.
Developed with Python, scikit-learn, Streamlit, and NLTK.
