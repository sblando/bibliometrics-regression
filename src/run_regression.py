import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# File paths used in the project
# ---------------------------------------------------------------------------
CLEAN_INPUT_PATH = "data/processed/researchers_clean.csv"
SCATTER_PLOT_PATH = "outputs/plots/scatter_documents_vs_citations.png"
REGRESSION_PLOT_PATH = "outputs/plots/regression_line.png"
SUMMARY_REPORT_PATH = "outputs/reports/regression_summary.txt"

# Ensure that output folders exist to prevent errors when saving files
os.makedirs(os.path.dirname(SCATTER_PLOT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_REPORT_PATH), exist_ok=True)


def _set_integer_xticks(series_with_integers: pd.Series) -> None:
    """
    Set X-axis ticks to integer values only.
    This is necessary because the number of documents is a whole number
    (you cannot have 3.5 papers).
    """
    min_value = int(np.nanmin(series_with_integers))
    max_value = int(np.nanmax(series_with_integers))
    plt.xticks(np.arange(min_value, max_value + 1, 1))


def _highlight_top3(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Highlight and label the top 3 most-cited researchers.

    Parameters:
        ax  - the current matplotlib axis where highlighting will be applied
        df  - the complete dataframe containing researcher data
    """
    if df.empty:
        return  # If the dataset is empty, there is nothing to highlight.

    # Select up to 3 researchers with the highest number of citations
    top3 = df.nlargest(min(3, len(df)), "total_citations")

    # Plot these top performers with a different color and larger markers
    ax.scatter(
        top3["document_count"],
        top3["total_citations"],
        s=120,               # larger marker size
        color="red",         # highlighted in red
        edgecolors="black",  # outlined for emphasis
        linewidths=0.6,
        label="Top 3 by citations",
        zorder=3,           # ensures this layer stays on top
    )

    # If researcher names are available, display them next to the highlighted points
    if "researcher_name" in df.columns:
        for _, row in top3.iterrows():
            ax.text(
                row["document_count"] + 0.05,   # small shift right so text doesn't overlap the point
                row["total_citations"] + 1.0,    # small shift upward for visibility
                str(row["researcher_name"]),     # convert to string in case of unexpected values
                fontsize=8,
                color="red",
                zorder=4,
            )


def main() -> None:
    """
    Main workflow:
    1. Load cleaned dataset
    2. Create a scatter plot
    3. Train a linear regression model
    4. Plot the regression line with highlighted top 3 researchers
    5. Save a text summary report of the model
    """
    # --- Step 1: Load cleaned data into a dataframe ---
    researcher_dataframe = pd.read_csv(CLEAN_INPUT_PATH)

    # Safety check: verify that the necessary columns exist
    required_columns = {"document_count", "total_citations"}
    missing = required_columns - set(researcher_dataframe.columns)
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    # --- Step 2: Create the scatter plot ---
    fig, ax = plt.subplots()
    ax.scatter(
        researcher_dataframe["document_count"],
        researcher_dataframe["total_citations"],
        label="Other researchers",
        alpha=0.9
    )

    # Call the function to highlight the top three researchers, if applicable
    _highlight_top3(ax, researcher_dataframe)

    # Set integer ticks for the X-axis
    _set_integer_xticks(researcher_dataframe["document_count"])

    # Add descriptive axis labels and title
    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Total Citations")
    ax.set_title("Scatter Plot: Documents vs Total Citations")

    # Apply a subtle dotted grid for visual clarity
    ax.grid(True, linestyle=":")

    # Display legend (shows top-3 highlighted category)
    ax.legend()

    # Save the completed scatter plot
    fig.savefig(SCATTER_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Step 3: Prepare data for linear regression ---
    # Reshape the independent variable (X) into a matrix: (n_samples, 1)
    document_count_matrix = researcher_dataframe[["document_count"]].values
    total_citations_vector = researcher_dataframe["total_citations"].values  # dependent variable (Y)

    # --- Step 4: Train the linear regression model ---
    linear_model = LinearRegression()
    linear_model.fit(document_count_matrix, total_citations_vector)

    # Generate predictions and compute model accuracy (R² score)
    fitted_values = linear_model.predict(document_count_matrix)
    r2_value = r2_score(total_citations_vector, fitted_values)

    # --- Step 5: Create a regression plot ---
    fig, ax = plt.subplots()

    # Replot original data
    ax.scatter(
        document_count_matrix,
        total_citations_vector,
        label="Other researchers",
        alpha=0.9
    )

    # Create a smooth line for the regression line
    x_min = float(document_count_matrix.min())
    x_max = float(document_count_matrix.max())
    x_grid = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_grid = linear_model.predict(x_grid)

    # Plot the regression line
    ax.plot(x_grid, y_grid, linestyle="--", label="Linear regression")

    # Highlight the top 3 researchers again in this plot
    _highlight_top3(ax, researcher_dataframe)

    # Set integer-based ticks for the X-axis
    _set_integer_xticks(researcher_dataframe["document_count"])

    # Add axis labels and title
    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Total Citations")
    ax.set_title("Regression Line: Documents vs Total Citations")
    ax.legend()
    ax.grid(True, linestyle=":")

    # Save regression plot
    fig.savefig(REGRESSION_PLOT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Step 6: Write a summary text report of the regression model ---
    with open(SUMMARY_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("Linear Regression: total_citations ~ document_count\n")
        report_file.write(f"Intercept (β0): {linear_model.intercept_:.6f}\n")
        report_file.write(f"Coefficient (β1): {linear_model.coef_[0]:.6f}\n")
        report_file.write(f"R-squared: {r2_value:.6f}\n")
        report_file.write("\nModel equation:\n")
        report_file.write(
            f"total_citations = {linear_model.intercept_:.6f} "
            f"+ {linear_model.coef_[0]:.6f} × document_count\n"
        )

    # --- Completion message for terminal users ---
    print("Regression complete.")
    print(f"Scatter saved:    {SCATTER_PLOT_PATH}")
    print(f"Regression saved: {REGRESSION_PLOT_PATH}")
    print(f"Report saved:     {SUMMARY_REPORT_PATH}")


if __name__ == "__main__":
    main()
