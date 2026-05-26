#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


def main():
    eval_dir = Path("results/evaluation")
    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_dir.exists():
        print(f"Eval directory {eval_dir} not found.")
        return

    data = []

    # Parse evaluation results
    for model_dir in eval_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for json_file in model_dir.glob("*_results.json"):
            dataset_name = json_file.stem.replace("_results", "")

            with open(json_file, "r") as f:
                content = json.load(f)

            for item in content.get("results", []):
                actual_mos = item.get("mos")
                predicted_mos = item.get("predicted_mos")

                if actual_mos is not None and predicted_mos is not None:
                    data.append(
                        {
                            "Model": model_name,
                            "Dataset": dataset_name,
                            "Actual MOS": actual_mos,
                            "Predicted MOS": predicted_mos,
                        }
                    )

    if not data:
        print("No evaluation data found with both Actual and Predicted MOS.")
        return

    df = pd.DataFrame(data)

    # 1. Compute aggregate metrics per model per dataset
    metrics = []
    for (model, dataset), group in df.groupby(["Model", "Dataset"]):
        y_true = group["Actual MOS"]
        y_pred = group["Predicted MOS"]

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        # Need at least 2 data points for correlation
        if len(y_true) > 1:
            pearson = pearsonr(y_true, y_pred)[0]
            spearman = spearmanr(y_true, y_pred)[0]
        else:
            pearson = float("nan")
            spearman = float("nan")

        metrics.append(
            {
                "Model": model,
                "Dataset": dataset,
                "Samples": len(y_true),
                "MAE": mae,
                "MSE": mse,
                "Pearson_r": pearson,
                "Spearman_rho": spearman,
            }
        )

    metrics_df = pd.DataFrame(metrics)

    # Print formatted metrics
    print("=== Evaluation Metrics ===")
    print(metrics_df.to_string(index=False, float_format="%.3f"))
    print("=========================\\n")

    # Save metrics to CSV
    metrics_csv = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to {metrics_csv}")

    # Set visualization style
    sns.set_theme(style="whitegrid")

    # 2. Generate Bar Plots for each metric
    for metric in ["MAE", "MSE", "Pearson_r", "Spearman_rho"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics_df, x="Dataset", y=metric, hue="Model")
        plt.title(f"Comparison of {metric} across Models & Datasets")
        plt.ylabel(metric)
        plt.xlabel("Dataset")
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plot_path = out_dir / f"barplot_{metric.lower()}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved bar plot: {plot_path}")

    # 3. Generate Scatter Plots (Actual vs Predicted MOS)
    # We will use sns.lmplot/FacetGrid to create a grid of scatter plots
    g = sns.FacetGrid(
        df, col="Dataset", row="Model", margin_titles=True, height=4, aspect=1
    )
    g.map(sns.scatterplot, "Actual MOS", "Predicted MOS", alpha=0.6, edgecolor=None)

    # Add y=x reference line
    for ax in g.axes.flat:
        ax.plot([1, 5], [1, 5], color="red", linestyle="--", linewidth=1.5, label="y=x")
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_aspect("equal", adjustable="box")

    # Optional: add legend for the y=x line
    if len(g.axes) > 0 and len(g.axes[0]) > 0:
        g.axes[0][0].legend(loc="upper left")

    g.fig.suptitle("Actual vs Predicted MOS by Model and Dataset", y=1.02)
    plt.tight_layout()
    scatter_path = out_dir / "scatter_actual_vs_predicted.png"
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    print(f"\\nAnalysis completed successfully! All outputs are in '{out_dir}/'")


if __name__ == "__main__":
    main()
