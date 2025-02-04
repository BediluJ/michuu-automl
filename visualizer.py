import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance

class Visualizer:
    plot_objects = []
    
    @classmethod
    def display_auto_eda(cls, df):
        """Main EDA visualization handler"""
        cls.plot_objects = []
        cls._plot_distributions(df)
        cls._plot_correlations(df)
        cls._plot_text_wordclouds(df)
        
    @classmethod
    def _plot_distributions(cls, df):
        """Plot numerical feature distributions"""
        num_cols = df.select_dtypes(include='number').columns
        for col in num_cols:
            try:
                fig = px.histogram(
                    df, 
                    x=col, 
                    title=f'Distribution of {col}',
                    marginal='box',
                    nbins=50
                )
                cls.plot_objects.append(fig)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting {col}: {str(e)}")
    
    @classmethod
    def _plot_correlations(cls, df):
        """Plot feature correlation matrix"""
        try:
            numeric_df = df.select_dtypes(include='number')
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    title="Feature Correlations",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                cls.plot_objects.append(fig)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting correlations: {str(e)}")
    
    @classmethod
    def _plot_text_wordclouds(cls, df):
        """Generate word clouds for text columns"""
        text_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in text_cols:
            try:
                text = ' '.join(df[col].dropna().astype(str))
                if len(text) > 0:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white'
                    ).generate(text)
                    
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    cls.plot_objects.append(fig)
                    st.pyplot(fig)
                    plt.close(fig)  # Clean up memory
            except Exception as e:
                st.error(f"Error generating wordcloud for {col}: {str(e)}")

    @classmethod
    def get_plot_objects(cls):
        """Retrieve stored plot objects for reporting"""
        return cls.plot_objects

    @staticmethod
    def plot_combined_roc(models, X_test, y_test):
        """Plot combined ROC curves for multiple models"""
        fig = go.Figure()
        
        for model in models:
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_test)
                    if probas.shape[1] == 2:  # Binary classification
                        fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
                    else:  # Multi-class (OvR)
                        fpr, tpr, _ = roc_curve(y_test, probas.ravel())
                    
                    roc_auc = auc(fpr, tpr)
                    model_name = model.__class__.__name__
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f'{model_name} (AUC = {roc_auc:.2f})',
                        mode='lines'
                    ))
            except Exception as e:
                st.error(f"ROC error for {model}: {str(e)}")
        
        fig.update_layout(
            title='Model ROC-AUC Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig)

    @staticmethod
    def show_model_diagnostics(model, X_test, y_test, problem_type):
        """Display model-specific diagnostics"""
        if problem_type == "classification":
            try:
                if not hasattr(model, 'predict_proba'):
                    st.warning("Model doesn't support probability predictions")
                    return None
                
                probas = model.predict_proba(X_test)
                
                # Multi-class handling
                if probas.shape[1] > 2:
                    fpr, tpr, _ = roc_curve(y_test, probas.ravel())
                else:  # Binary classification
                    fpr, tpr, _ = roc_curve(y_test, probas[:, 1])
                
                roc_auc = auc(fpr, tpr)

                fig = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
                )
                fig.add_shape(
                    type='line',
                    line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1
                )
                st.plotly_chart(fig)
                return fig
            except Exception as e:
                st.error(f"Diagnostics error: {str(e)}")
                return None
        return None

    @staticmethod
    def show_feature_importance(model, feature_names, X_test, y_test, max_features=10):
        """Display feature importance with permutation importance fallback"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                result = permutation_importance(
                    model, X_test, y_test,
                    n_repeats=10,
                    random_state=42,
                    n_jobs=-1
                )
                importances = result.importances_mean
                
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(max_features)
            
            fig = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top Feature Importance',
                labels={'Importance': 'Importance Score'}
            )
            st.plotly_chart(fig)
            return fig
        except Exception as e:
            st.error(f"Feature importance error: {str(e)}")
            return None