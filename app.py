import streamlit as st
import pandas as pd
from EasyML.data import read_file

def main():
    # App name
    st.title("Easy ML")

    # File uploader area
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file into a df
        df = read_file(uploaded_file)

        # Display a preview of the data
        st.dataframe(df.head())

        st.write("Null Value Counts")
        st.dataframe(df.isna().sum().rename_axis("Column").rename("Null Counts"))

        # Select feature columns
        feature_column = st.multiselect(
            'Select feature column:',
            df.columns,
            default=df.columns  # Select all columns by default
        )

        # Select target columns
        target_column = st.selectbox("Select Target Column", options=df.columns)

        # Task type selection
        task_type = st.selectbox("Select Task Type", options=["Regression", "Classification"])

        # Model selection based on task type
        if task_type == "Regression":
            # model_choice = st.selectbox("Select Regression Model", options=list(regression_models.keys()))
            pass
        else:
            # model_choice = st.selectbox("Select Classification Model", options=list(classification_models.keys()))
            pass

        # Train model button
        if st.button("Train Model"):
            # Add your training logic here
            # For example: model = ...; model.fit(X, y)
            st.write("Training model...")
            pass


if __name__ == "__main__":
    main()
