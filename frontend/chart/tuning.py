import optuna
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from sklearn.metrics import confusion_matrix
import optuna.visualization as vis
from backend.tuning import evaluate_model


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        colorscale='Viridis',
        showscale=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted labels',
        yaxis_title='True labels',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def customize_plotly_fig(fig):
    # Apply Viridis colormap and adjust other properties
    for trace in fig.data:
        if 'colorscale' in trace:
            trace.colorscale = 'Viridis'
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def display_study_results(study):
    best_params = study.best_params
    folds = st.session_state.logo_folds

    # Display the optimization history and parameter importances for the first fold
    fig = vis.plot_optimization_history(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    fig = vis.plot_param_importances(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    fig = vis.plot_parallel_coordinate(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    fig = vis.plot_slice(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    fig = vis.plot_edf(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    # Contour Plot with SelectBox for X and Y axes
    @st.experimental_fragment
    def contour_plot_fragment():
        param_names = list(best_params.keys())
        x_param = st.selectbox("Select X-axis parameter for Contour Plot", param_names)
        y_param = st.selectbox("Select Y-axis parameter for Contour Plot", param_names, index=2)
        if x_param != y_param:
            fig = vis.plot_contour(study, params=[x_param, y_param])
            st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)
            st.write(
                f"The contour plot visualizes the relationship between {x_param} and {y_param}, showing how these two parameters interact and affect the objective value.")
        else:
            st.write("Please select different parameters for X and Y axes.")

    contour_plot_fragment()

    fig = vis.plot_rank(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

    fig = vis.plot_timeline(study)
    st.plotly_chart(customize_plotly_fig(fig), use_container_width=True)

