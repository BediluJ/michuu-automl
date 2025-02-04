from jinja2 import Environment, FileSystemLoader
import base64
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.io import to_html
import logging

class ReportGenerator:
    @staticmethod
    def generate_report(df_info, eda_plots, model_info):
        """
        Generates a comprehensive HTML report from analysis results
        Args:
            df_info: Dictionary of dataset metadata
            eda_plots: List of EDA visualization figures
            model_info: Dictionary containing model evaluation data
        Returns:
            Rendered HTML report as string
        """
        env = Environment(loader=FileSystemLoader('templates/'))
        template = env.get_template('report_template.html')
        
        processed_data = {
            'plot_html': [],
            'plot_images': [],
            'diagnostics': {}
        }

        # Process diagnostic plots
        if 'diagnostics' in model_info:
            processed_data['diagnostics'] = ReportGenerator._process_figures(
                model_info['diagnostics'],
                "Diagnostic"
            )

        # Process EDA plots
        eda_results = ReportGenerator._process_figures(eda_plots, "EDA")
        processed_data['plot_html'] = eda_results['html']
        processed_data['plot_images'] = eda_results['images']

        return template.render(
            df_info=df_info,
            model_info=model_info,
            **processed_data
        )

    @staticmethod
    def _process_figures(figures, plot_type="Figure"):
        """
        Unified figure processing for both Matplotlib and Plotly figures
        Returns dict with 'html' and 'images' keys
        """
        result = {'html': [], 'images': []}
        
        if isinstance(figures, dict):
            # Handle diagnostic figures dictionary
            for name, fig in figures.items():
                try:
                    if isinstance(fig, go.Figure):
                        result['html'].append(to_html(fig, full_html=False))
                    elif isinstance(fig, plt.Figure):
                        result['images'].append(ReportGenerator._fig_to_base64(fig))
                except Exception as e:
                    logging.error(f"Error processing {plot_type} plot {name}: {str(e)}")
        else:
            # Handle EDA plots list
            for idx, fig in enumerate(figures):
                try:
                    if isinstance(fig, go.Figure):
                        result['html'].append(to_html(fig, full_html=False))
                    elif isinstance(fig, plt.Figure):
                        result['images'].append(ReportGenerator._fig_to_base64(fig))
                except Exception as e:
                    logging.error(f"Error processing {plot_type} plot #{idx}: {str(e)}")
        
        return result

    @staticmethod
    def _fig_to_base64(fig):
        """Convert matplotlib figure to base64 encoded PNG"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        plt.close(fig)  # Critical for memory management
        return encoded