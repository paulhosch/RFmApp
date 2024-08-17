from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

from backend.ee import create_average_absolute_correlation_matrix

def plot_3d_correlation_scatter_with_heatmap(correlations, features, title, correlation_type, groups):
    """
    Plots a 3D scatter chart of correlations with an average absolute correlation heatmap.

    Args:
        correlations (dict): Dictionary with group labels as keys and lists of correlation data as values.
        features (list): List of feature names.
        title (str): Title of the plot.
        correlation_type (str): Type of correlation (e.g., "Pearson's" or "Spearman's").
        groups (list): List of group labels.
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "scene"}, {"type": "heatmap"}]],
        subplot_titles=["3D Scatter Plot", "Average Absolute Correlation Matrix"]
    )

    group_labels = [group['label'] for group in groups]
    viridis_colors = px.colors.sample_colorscale('Viridis', [i / (len(group_labels) - 1) for i in range(len(group_labels))])
    group_colors = {label: viridis_colors[i] for i, label in enumerate(group_labels)}

    # 3D Scatter Plot
    for group_label, corr_data in correlations.items():
        x = [data[0] for data in corr_data]
        y = [data[1] for data in corr_data]
        z = [data[2] for data in corr_data]

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=group_colors[group_label], opacity=0.8),
            name=group_label,
            text=[f"{x[i]}, {y[i]}: {z[i]:.2f}" for i in range(len(x))],
            hoverinfo="text+name"
        ), row=1, col=1)

    # Average Absolute Correlation Heatmap
    avg_abs_corr_matrix = create_average_absolute_correlation_matrix(correlations, features)

    fig.add_trace(go.Heatmap(
        z=avg_abs_corr_matrix,
        x=features,
        y=features,
        colorscale='Viridis',
        zmin=0, zmax=1,  # Absolute correlation ranges from 0 to 1
        texttemplate="%{z:.2f}",
        textfont={"size": 16},
    ), row=1, col=2)

    fig.update_layout(
        title=f"{correlation_type} Correlation Analysis",
        scene=dict(
            xaxis_title='Feature',
            yaxis_title='Feature',
            zaxis_title='Correlation',
            xaxis=dict(tickmode='array', tickvals=list(range(len(features))), ticktext=features),
            yaxis=dict(tickmode='array', tickvals=list(range(len(features))), ticktext=features),
            zaxis=dict(range=[-1, 1])
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ),
        legend_title="Observation Groups",
        height=700,
        width=1200,
    )

    return fig