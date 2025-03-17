import pandas as pd
import streamlit as st
from EasyML import data as data
from EasyML.models import (Engine, classification_models, regression_models, regression_metrics, classification_metrics,
                           scalers, encoders)
from EasyML.AutoML import AutoML


def save_model_callback(engine):
    """Saves the model in session state to allow downloading"""
    model_save_path = engine.save_model()

    with open(model_save_path, "rb") as f:
        st.session_state["model_bytes"] = f.read()

def main():
    # App name
    st.set_page_config(page_title="Easy-ML")
    st.title("Easy ML")

    # File uploader area
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Read the file into a df
        df = data.read_file(uploaded_file)
        st.markdown("---")

        # Select target columns
        target_column = st.selectbox("Select Target Column (Selects Last Column by Default)", options=df.columns, index = len(df.columns)-1)
        st.session_state['target_column'] = target_column # Used session state.
        st.markdown("---")

        try:
            # Verify there are no null values in target column
            data.check_target_column_null_values(df, target_column)
        except KeyError as e:
            st.write(str(e))
            return
        except ValueError as e:
            st.write(str(e))
            return

        # Display a preview of the data
        st.subheader("Dataframe Head")
        st.dataframe(df.head())
        st.markdown("---")

        col1, col2 = st.columns(2)
        if col1.button("EDA"):
            st.session_state['display_mode'] = 'EDA'
        if col2.button("Train Model"):
            st.session_state['display_mode'] = 'Train Model'

        if st.session_state.get('display_mode') == "EDA":
            st.subheader("Exploratory Data Analysis (EDA)")
            
            # Summary statistics subsection
            summary_stats = data.get_summary_statistics(df)

            st.subheader("Numeric Statistics Summary")
            if not summary_stats['numeric'].empty:
                st.dataframe(summary_stats['numeric'])
            else:
                st.write("No Numeric features to summarize.")

            st.subheader("Categorical Summary Statistics")
            if not summary_stats['categorical'].empty:
                st.dataframe(summary_stats['categorical'])
            else:
                st.write("No Categorical Features to summarize.")
            st.markdown("---")

            # Null values sub-section
            st.subheader("Null Value Counts")
            st.dataframe(df.isna().sum().rename_axis("Column").rename("Null Counts"))

            # Imputation recommendation subsection
            st.subheader("Missing Value Imputation Recommendation")
            imputation_recs = data.recommend_imputation(df, target_column=target_column)

            # Numeric Columns
            st.write("**Numeric Columns**")
            if imputation_recs['Numeric']['columns']:
                numeric_rec_df = pd.DataFrame(
                    list(imputation_recs['Numeric']['columns'].items()),
                    columns= ['Columns','Recommended Imputation']
                )
                st.table(numeric_rec_df)

                # Majority Vote + Implications
                st.write("Majority Vote:", imputation_recs['Numeric']['majority_recommendation'])
                st.write(imputation_recs['Numeric']['implications'])
            else:
                st.write("No Numeric Features Detected.")

            st.write("**Categorical Columns**")
            if imputation_recs["Categorical"]["columns"]:
                categorical_rec_df = pd.DataFrame(
                    list(imputation_recs["Categorical"]["columns"].items()),
                    columns=["Columns", "Recommended Imputation"]
                )
                st.table(categorical_rec_df)

                # Majority vote + implications
                st.write("**Majority Recommendation**:", imputation_recs["Categorical"]["majority_recommendation"])
                st.write(imputation_recs["Categorical"]["implications"])
            else:
                st.write("No Categorical Features Detected.")

            st.markdown("---")

        # Model training section
        elif st.session_state.get('display_mode') == "Train Model":
            st.subheader("Train Model")

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
                st.write("Null value check for categorical columns")
                categorical_imputation_method = st.selectbox(
                    "Select an option",
                    options=data.categorical_impute_options
                )

                categorical_value = ""
                if categorical_imputation_method == "Custom Value":
                    value = st.text_input("Enter the custom value to impute")

            # Run imputation of null values
            df = data.impute(df, numeric_imputation_method, "Numeric", value=numeric_value)
            df = data.impute(df, categorical_imputation_method, "Categorical", category=categorical_value)
            
            st.markdown("---")
            
            # Select feature columns
            default_features = [col for col in df.columns if col != target_column]
            feature_columns = st.multiselect(
                'Select feature column:',
                df.columns,
                default=default_features  # Updated Code to not show target column by default
            )

            st.markdown("---")

            # Get training size
            train_size = st.number_input("Training Size:", min_value=0.0, max_value=1.0, value=0.8)

            # Get shuffle parameter
            shuffle = st.selectbox("Shuffle Data For Training:", ["Yes", "No"])
            shuffle = True if shuffle == "Yes" else False

            # Get input for scaler and encoder
            col1, col2 = st.columns(2)

            with col1:
                scaler = st.selectbox("Select Scaler", scalers)

            with col2:
                encoder = st.selectbox("Select Encoder", encoders)
            
            st.markdown("---")

            # Task type selection
            task_type = st.selectbox("Select Task Type", options=["Regression", "Classification"])

            # Model selection based on task type
            if task_type == "Regression":
                model_choice = st.selectbox("Select Regression Model", options=regression_models)
                metric_type = st.selectbox("Select Evaluation Metric", options=regression_metrics)
            else:
                model_choice = st.selectbox("Select Classification Model", options=classification_models)
                metric_type = st.selectbox("Select Evaluation Metric", options=classification_metrics)

            st.markdown("---")

            # Train model button
            col1 , col2 = st.columns(2)
            button = ""
            training_completed = False

            if "engine" not in st.session_state:
                st.session_state["engine"] = None

            if "model_bytes" not in st.session_state:
                st.session_state["model_bytes"] = None

            with col1:
                if st.button("Train selected model"):
                    button = "Standard ML"

            # Auto ML button
            with col2:
                if st.button("Auto ML"):
                    button = "Auto ML"

            # Training pipelines
            if button == "Standard ML":
                with st.spinner("Training the model ..."):
                    engine = Engine(model=model_choice,
                                    data=df,
                                    features=feature_columns,
                                    target=target_column,
                                    task=task_type,
                                    scaler=scaler,
                                    encoder=encoder,
                                    metric=metric_type,
                                    train_size=train_size,
                                    shuffle=shuffle)

                    engine.train()
                    results = engine.get_metrics()
                    training_completed = True
                    st.session_state["engine"] = engine
            elif button == "Auto ML":
                with st.spinner("Training Auto ML pipeline ..."):
                    engine = AutoML(data=df,
                                        feature_cols=feature_columns,
                                        target=target_column,
                                        train_size=train_size)

                    engine.train()
                    results = engine.get_leaderboard()
                    training_completed = True
                    st.session_state["engine"] = engine

            st.markdown("---")

            # Results subsection
            if training_completed:
                st.success("Training completed")
                st.write("Results")

                if isinstance(results, float):
                    st.write(f"{metric_type}: {results:.2f}")
                else:
                    st.dataframe(results)

                download_label = "Download Trained Model" if button == "Standard ML" else "Download Best Model"
                engine = st.session_state["engine"]

                st.download_button(
                    label=download_label,
                    data=st.session_state["model_bytes"] if st.session_state["model_bytes"] else b"",
                    file_name="model",
                    mime="application/octet-stream",
                    on_click=save_model_callback,
                    args=[engine],
                )

                st.markdown("---")


if __name__ == "__main__":
    main()
