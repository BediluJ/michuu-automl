# main_app.py
import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from visualizer import Visualizer
from model_builder import ModelBuilder
from report_generator import ReportGenerator

def main():
    st.set_page_config(page_title="👌Michuu AutoML", layout="wide")
    st.title(" 👌Michuu AutoML")
    "*Before uploading a dataset, please ensure that your data is properly deidentified and does not contain any sensitive information."
    # File upload with format support
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Immediate target selection with validation
            target = st.selectbox("Select Target Variable", df.columns, key='target_select')
            
            if df[target].nunique() == 1:
                st.error("Target variable must have more than one unique value")
                return

            processor = DataProcessor(df, target)
            
            # Advanced options sidebar
            st.sidebar.header("Advanced Configuration")
            balance_method = st.sidebar.selectbox(
                "Class Imbalance Handling",
                ["None", "SMOTE", "Class Weight Adjustment"]
            )
            # Data sampling option
            sample_size = st.sidebar.slider("Training Sample %", 10, 100, 100)
            
            # Model optimization options
            use_cv = st.sidebar.checkbox("Enable Cross-Validation")
            cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5) if use_cv else None
            tune_params = st.sidebar.checkbox("Enable Hyperparameter Tuning")
            tune_method = st.sidebar.selectbox(
                "Tuning Method", 
                ["GridSearch", "RandomSearch"]
            ) if tune_params else None
            
            with st.spinner('Processing data...'):
                cleaned_df = processor.process_data(balance_method=balance_method)
            
             # Apply sampling after processing
                if sample_size < 100:
                    st.info(f"Using {sample_size}% sample ({len(cleaned_df)*sample_size//100} records)")
                    train_df = cleaned_df.sample(frac=sample_size/100, random_state=42)
                else:
                    train_df = cleaned_df.copy()
            
            
            # Descriptive statistics section
            st.header("Data Overview")
            stats = processor.get_descriptive_stats()
            st.dataframe(
                stats.style.format("{:.2f}", na_rep="N/A"),
                use_container_width=True,
                height=400
            )
            
            # EDA Visualizations
            st.header("Exploratory Analysis")
            Visualizer.display_auto_eda(cleaned_df)
            
            # Model training section
            st.header("Automated Machine Learning")
            with st.spinner('Training models...'):
                model_builder = ModelBuilder(
                    cleaned_df, 
                    target,
                    use_cv=use_cv,
                    cv_folds=cv_folds,
                    tune_params=tune_params,
                    tune_method=tune_method
                )
                models_df = model_builder.compare_models()
                
                # Get performance metric
                metric = 'AUC_ROC' if model_builder.problem_type == 'classification' else 'R2'
                
                # Sort models by performance metric
                sorted_df = models_df.sort_values(metric, ascending=False).reset_index(drop=True)
                
                # Model comparison and selection
                st.subheader("Model Performance Comparison")
                
                # Highlight champion row
                def highlight_champion(row):
                    return ['background-color: lightgreen' if row.name == 0 else '' for _ in row]
                
                # Display sorted and highlighted dataframe
                numeric_cols = sorted_df.select_dtypes(include=[np.number]).columns
                st.dataframe(
                    sorted_df.style
                        .format("{:.3f}", subset=numeric_cols)
                        .apply(highlight_champion, axis=0),
                    use_container_width=True,
                    height=400
                )
                
                # Get and display champion model
                champion_model = model_builder.get_champion_model()
                
                st.success(f"🏆 Champion Model: {champion_model['Model']}")
                st.metric(
                    f"Best {metric.replace('_', ' ')} Score",
                    f"{champion_model['Score']:.3f}",
                    help=f"Based on {metric.replace('_', ' ')} metric"
                )
                
                # Model diagnostics
                if model_builder.problem_type == 'classification':
                    st.subheader("Model ROC-AUC Comparison")
                    Visualizer.plot_combined_roc(
                        model_builder.models,
                        model_builder.X_test,
                        model_builder.y_test
                    )
                
                # Feature importance visualization
                st.subheader("Feature Importance")
                Visualizer.show_feature_importance(
                    model_builder.best_model,
                    model_builder.X.columns,
                    model_builder.X_test,
                    model_builder.y_test
                )
                
                # Report generation
                with st.spinner('Compiling report...'):
                    report = ReportGenerator.generate_report(
                        df_info=processor.get_data_info(),
                        eda_plots=Visualizer.get_plot_objects(),
                        model_info={
                            'champion_model': champion_model,
                            'all_models': sorted_df,
                            'problem_type': model_builder.problem_type,
                            'optimization_params': {
                                'cv_used': use_cv,
                                'cv_folds': cv_folds,
                                'tuning_method': tune_method if tune_params else None
                            }
                        }
                    )
                    
                    st.download_button(
                        "📥 Download Full Report",
                        data=report,
                        file_name="automl_report.html",
                        mime="text/html",
                        help="Download comprehensive report with all analysis details"
                    )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()

if __name__ == "__main__":
    main()
