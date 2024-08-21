import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns



def plot_importances(importance_proxies, impurity_importances=None, permutation_importances=None,
                     aggregated_shap_values=None, aggregated_X_test=None, feature_names=None):
    if 'Shapley' in importance_proxies and aggregated_shap_values is not None and aggregated_X_test is not None:
        st.write('##### Shapley Value based Feature Importance')
        shap.summary_plot(aggregated_shap_values[:, :, 1], aggregated_X_test)
        st.pyplot(plt.gcf())
        plt.clf()

    if 'Impurity Reduction' in importance_proxies and impurity_importances is not None:
        st.write('##### Mean Decrease in Gini Impurity based Feature Importance')
        fig, ax = plt.subplots(figsize=(10, 8))
        df = pd.DataFrame(impurity_importances)

        # Melt the dataframe for seaborn
        df_melted = df.melt(var_name='Feature', value_name='Importance')

        # Create horizontal violin plot
        sns.violinplot(x='Importance', y='Feature', data=df_melted, ax=ax, orient='h')

        # Customize the plot
        ax.set_xlabel("Importance", fontsize=12)

        # Set transparent background
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        plt.clf()

    if 'Permutation Accuracy' in importance_proxies and permutation_importances is not None:
        st.write("##### Feature Permutation Accuracy Decrease based Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        df = pd.DataFrame(permutation_importances)

        # Melt the dataframe for seaborn
        df_melted = df.melt(var_name='Feature', value_name='Importance')

        # Create horizontal violin plot
        sns.violinplot(x='Importance', y='Feature', data=df_melted, ax=ax, orient='h')

        # Customize the plot
        ax.set_xlabel("Importance", fontsize=12)

        # Set transparent background
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        plt.clf()