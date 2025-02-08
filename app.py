import streamlit as st
import pandas as pd
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, predict_model as reg_predict, pull as reg_pull
from pycaret.classification import setup as clf_setup, compare_models as clf_compare, predict_model as clf_predict, pull as clf_pull

st.title("AutoML App using PyCaret")
# Initialize session state for df if not already set
if "df" not in st.session_state:
    st.session_state["df"] = None


# File Upload
file_upload = st.file_uploader("Upload a CSV file", type=["csv"])

if file_upload is not None:
    try:
        df = pd.read_csv(file_upload)
        if df.empty:
            st.error("The uploaded file is empty.")
        else:
            st.write("### Data Preview", df.head())
            st.session_state["df"] = df
    except Exception as e:
        st.error(f"Error reading file: {e}")
    
else:
    st.warning("Please upload a CSV file to proceed.")

df=st.session_state["df"]
# Select Problem Type
problem_type = st.selectbox("Choose problem type", ["Regression", "Classification"])

# # Ensure at least one column exists
# if df.shape[1] == 0:
#     st.error("The uploaded file has no valid columns. Please check the file and try again.")

target = st.selectbox("Select target column", df.columns)

# Train Button
if st.button("Train Model"):
    with st.spinner("Training in progress... Please wait."):
        try:
            if problem_type == "Regression":
                st.write("### Running Regression Model Training")
                s = reg_setup(data=df, target=target, session_id=123, silent=True)
                best_model = reg_compare()
                st.write("### Best Model:", best_model)

                # Show Model Performance Summary
                metrics = reg_pull()
                st.write("### Model Performance Metrics")
                st.write(metrics)

                pred_holdout = reg_predict(best_model)

            elif problem_type == "Classification":
                st.write("### Running Classification Model Training")
                s = clf_setup(data=df, target=target, session_id=123, silent=True)
                best_model = clf_compare()
                st.write("### Best Model:", best_model)

                # Show Model Performance Summary
                metrics = clf_pull()
                st.write("### Model Performance Metrics")
                st.write(metrics)

                pred_holdout = clf_predict(best_model)

            st.write("### Predictions on Holdout Set", pred_holdout.head())

            # Provide option to download predictions
            csv = pred_holdout.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"An error occurred during model training: {e}")
