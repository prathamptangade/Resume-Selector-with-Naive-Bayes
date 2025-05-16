import streamlit as st
from resume_selector import load_and_process_data, train_model, plot_confusion_matrix
import pandas as pd

st.title("Resume Selector with Naive Bayes")
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        X, y = load_and_process_data(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(pd.read_excel(uploaded_file, engine='openpyxl').head())

        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
        if st.button("Run Model"):
            with st.spinner("Training model..."):
                cm, report, y_pred, y_test = train_model(X, y, test_size=test_size)
                st.header("Model Results")
                st.subheader("Confusion Matrix")
                cm_img = plot_confusion_matrix(cm)
                st.image(f"data:image/png;base64,{cm_img}")
                st.subheader("Classification Report")
                st.json(report)
                st.subheader("Predictions")
                results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.dataframe(results)
                st.write(f"**Accuracy**: {report['accuracy']:.2%}")
    except Exception as e:
        st.error(f"Error processing file: {e}")
