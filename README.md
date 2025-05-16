# 🧠 Resume Selector – NLP Classification with Naive Bayes

This project implements a **Resume Classification System** using **Natural Language Processing (NLP)** and a **Naive Bayes classifier** to categorize resumes as `flagged` or `not flagged`. It includes:

- 📓 A **Jupyter Notebook** for training and analyzing the model
- 🌐 A **Streamlit Web Application** for interactive resume classification with dataset uploads

---

## 🚀 Features

- Load and process resume text data from Excel
- Clean invalid entries and handle missing data
- Convert resume text to numerical features using `CountVectorizer`
- Train a **Multinomial Naive Bayes** classifier
- Visualize performance using a **confusion matrix** and **classification report**
- Deploy an interactive **Streamlit app** for real-time model evaluation

---

## 📁 File Structure

resume-selector/
├── app.py # Streamlit app interface
├── resume_selector.py # Core logic: data loading, model training, plotting
├── resume_selector.ipynb # Jupyter notebook version of model
├── sample_dataset.xlsx # Example input dataset
└── README.md # This documentation file

yaml
Copy
Edit

---

## 📦 Installation

Install all required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk openpyxl streamlit
📊 Input Dataset Requirements
Upload an Excel file (.xlsx) with at least the following columns:

resume_text – The text content of each resume

class – The label/class for the resume (e.g., flagged or not flagged)

Sample:

resume_text	class
"Experienced in Python…"	flagged
"Project management…"	not flagged

🧪 Using the Jupyter Notebook
Open resume_selector.ipynb in Jupyter and follow the steps to:

Load and clean the dataset

Vectorize the resume text

Train a Naive Bayes model

Generate metrics like:

Confusion Matrix

Accuracy

Classification Report

🌐 Running the Streamlit Web App
Start the app using:

bash
Copy
Edit
streamlit run app.py
What the App Does:
📂 Upload .xlsx file containing resumes and labels

🔧 Select test split ratio using slider

⚙️ Click “Run Model” to:

Train Naive Bayes classifier

Show confusion matrix as heatmap

Display classification report

Output predictions (actual vs. predicted)

Report overall accuracy

🛠 Code Overview
resume_selector.py
Contains 3 core functions:

load_and_process_data(file)

Cleans and vectorizes text data from Excel

train_model(X, y, test_size)

Splits dataset and trains a Naive Bayes model

plot_confusion_matrix(cm)

Creates a heatmap plot and returns base64 PNG

app.py
Implements the Streamlit interface

Handles file upload, model training, visualization

📈 Example Output (Web App)
✅ Confusion Matrix Heatmap

✅ Classification Report (precision, recall, f1-score)

✅ Accuracy Score

✅ Data preview & prediction table

🔮 Future Enhancements
Add support for TF-IDF or transformer-based encodings

Visualize misclassified examples

Export predictions to downloadable file

Support multi-class classification

Cloud deployment on Streamlit Community Cloud / Heroku

👨‍💻 Author
Developed with Python, scikit-learn, and Streamlit

📄 License
This project is licensed under the MIT License.

vbnet
Copy
Edit

Let me know if you'd like me to insert badges (e.g., for Python version or Streamlit app link), add screenshots, or adapt this README for a GitHub repo.







