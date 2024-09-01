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


def display_study_results(study):
    best_params = study.best_params
    folds = st.session_state.logo_folds

    # Display the optimization history
    try:
        fig = vis.plot_optimization_history(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot optimization history: {str(e)}")

    # Try to plot parameter importances, handle potential errors
    try:
        fig = vis.plot_param_importances(study)
        st.plotly_chart(fig, use_container_width=True)
    except RuntimeError as e:
        st.warning(f"Could not plot parameter importances: {str(e)}")

    # Try to plot parallel coordinate
    try:
        fig = vis.plot_parallel_coordinate(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot parallel coordinate: {str(e)}")

    # Try to plot slice
    try:
        fig = vis.plot_slice(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot slice: {str(e)}")

    # Try to plot EDF
    try:
        fig = vis.plot_edf(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot EDF: {str(e)}")

    # Contour Plot with SelectBox for X and Y axes
    @st.fragment
    def contour_plot_fragment():
        col1, col2 = st.columns([1,3])
        param_names = list(best_params.keys())
        x_param = col1.selectbox("Select X-axis parameter for Contour Plot", param_names)
        y_param = col1.selectbox("Select Y-axis parameter for Contour Plot", param_names, index=min(1, len(param_names)-1))
        if x_param != y_param:
            try:
                fig = vis.plot_contour(study, params=[x_param, y_param])
                col2.plotly_chart(fig, use_container_width=True)
                col2.write(
                    f"The contour plot visualizes the relationship between {x_param} and {y_param}, showing how these two parameters interact and affect the objective value.")
            except Exception as e:
                col2.warning(f"Could not plot contour: {str(e)}")
        else:
            col2.write("Please select different parameters for X and Y axes.")

    contour_plot_fragment()

    # Try to plot rank
    try:
        fig = vis.plot_rank(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot rank: {str(e)}")

    # Try to plot timeline
    try:
        fig = vis.plot_timeline(study)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not plot timeline: {str(e)}")