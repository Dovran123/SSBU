import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

class Plotter:
    """A class for plotting the results of machine learning experiments."""

    def __init__(self, output_dir):
        """
        Initialize the Plotter with the output directory for saving plots.

        Parameters:
        - output_dir: Directory where the plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        self.plot_counter = 0
    def plot_metric_density(self, results, metrics=('accuracy', 'f1_score', 'roc_auc')):
        """
        Plot density plots for specified metrics.

        Parameters:
        - results: DataFrame containing the results.
        - metrics: List of metrics to plot.
        """
        fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(20, 5))
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for label, df in results.groupby('model'):
                sns.kdeplot(data=df[metric], ax=ax, label=label)
            ax.set_title(f'Density plot of {metric.capitalize()}')
            ax.set_xlabel(metric.capitalize())
            ax.set_xlim(xmax=1)
            if i == 0:
                ax.legend()
        plt.tight_layout()
        self.save_plot()


    def plot_evaluation_metric_over_replications(self, all_metric_results, title, metric_name):
        """
        Plot accuracies for each model over all replications.

        Parameters:
        - all_accuracies: Dict containing accuracies for each model.
        """
        plt.figure(figsize=(10, 5))
        colors = ['green', 'orange', 'blue']  # Extend the color list as needed
        for i, (model_name, values) in enumerate(all_metric_results.items()):
            plt.plot(values, label=f"{model_name} per replication", alpha=0.5, color=colors[i % len(colors)])
            plt.axhline(y=np.mean(values), color=colors[i % len(colors)], linestyle='--',
                        label=f"{model_name} average")
        plt.title(title)
        plt.xlabel('Replication')
        plt.ylabel(metric_name)
        plt.legend()
        self.save_plot()

    def plot_confusion_matrices(self, confusion_matrices):
        """
        Plot the average confusion matrix for each model.

        Parameters:
        - confusion_matrices: Dict containing the average confusion matrix for each model.
        """
        for model_name, matrix in confusion_matrices.items():
            plt.figure(figsize=(6, 5))
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', cbar=False)
            plt.title(f'Average Confusion Matrix: {model_name}')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            self.save_plot()

    def print_best_parameters(self, results):
        """
        Print the most frequently chosen best parameters for each model.

        Parameters:
        - results: DataFrame containing the results.
        """
        for model_name in results['model'].unique():
            model_results = results[results['model'] == model_name]
            best_params_list = model_results['best_params'].value_counts().index[0]
            print(f"Most frequently chosen best parameters for {model_name}: {best_params_list}")

    def save_plot(self):
        # Save the plot to a file
        plot_path = os.path.join(self.output_dir, f'plot_{self.plot_counter}.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to release memory
        self.plot_counter += 1  # Increment the plot counter