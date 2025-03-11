import streamlit as st
import pandas as pd
from EasyML import data
from EasyML.models import Engine, classification_models, regression_models

def main():
    # App name
    st.title("Easy ML")

    # File uploader area
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file into a df
        df = data.read_file(uploaded_file)

        # Display a preview of the data
        st.dataframe(df.head())

        st.write("Null Value Counts")
        st.dataframe(df.isna().sum().rename_axis("Column").rename("Null Counts"))

        col1, col2 = st.columns(2)

        # Display options to deal with nulls in numeric columns
        with col1:
            st.write("Null value check for numeric columns")
            numeric_imputation_method = st.selectbox(
                                            "Select an option",
                                            options=data.numeric_impute_options
                                        )

            numeric_value = 0.0
            if numeric_imputation_method == "Custom Value":
                numeric_value = st.number_input("Enter the custom value to impute")

        # Display options to deal with nulls in categorical columns
        with col2:
            st.write("Null value check for numeric columns")
            categorical_imputation_method = st.selectbox(
                "Select an option",
                options=data.categorical_impute_options
            )

            categorical_value = 0.0
            if categorical_imputation_method == "Custom Value":
                value = st.text_input("Enter the custom value to impute")

        df = data.impute(df, numeric_imputation_method, "Numeric", value=numeric_value)
        df = data.impute(df, categorical_imputation_method, "Categorical", value=categorical_value)

        # Select feature columns
        feature_columns = st.multiselect(
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
            model_choice = st.selectbox("Select Regression Model", options=regression_models)
            pass
        else:
            model_choice = st.selectbox("Select Classification Model", options=classification_models)

        engine = Engine(model_choice, df, target_column)
        engine.set_features(feature_columns)

        # Train model button
        col1 , col2 = st.columns(2)

        with col1:
            if st.button("Train Model"):
                st.write("Training model...")
                engine.train()

        with col2:
            if st.button("Auto ML"):
                st.write("Finding best model nad parameters")

        if engine.is_trained():
            results = engine.get_metrics()
            st.dataframe(results)


if __name__ == "__main__":
    main()
