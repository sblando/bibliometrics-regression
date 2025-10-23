import os
import pandas as pd

# ---------- Paths ----------
RAW_INPUT_PATH = "data/raw/researchers.csv"            # Cambia el nombre si tu CSV se llama distinto
CLEAN_OUTPUT_PATH = "data/processed/researchers_clean.csv"

# Asegura que exista la carpeta de salida
os.makedirs(os.path.dirname(CLEAN_OUTPUT_PATH), exist_ok=True)

def main() -> None:
    # --- Load ---
    raw_dataframe = pd.read_csv(RAW_INPUT_PATH)

    # --- Normalize column names (strip/whitespace) ---
    raw_dataframe.columns = [c.strip() for c in raw_dataframe.columns]

    # --- Rename to meaningful names (ajusta si tu CSV difiere) ---
    column_mapping = {
        "Name": "researcher_name",
        "Web of Science Documents": "document_count",
        "Times Cited": "total_citations",
        "Category Normalized Citation Impact": "cnci",
        "H-Index": "h_index",
        "% All Open Access Documents": "pct_open_access",
    }
    standardized_dataframe = raw_dataframe.rename(columns=column_mapping)

    # --- Coerce numeric types (strings con %, etc.) ---
    numeric_columns = ["document_count", "total_citations", "cnci", "h_index", "pct_open_access"]
    for column_name in numeric_columns:
        if column_name in standardized_dataframe.columns:
            standardized_dataframe[column_name] = (
                standardized_dataframe[column_name]
                .astype(str)
                .str.replace("%", "", regex=False)
            )
            standardized_dataframe[column_name] = pd.to_numeric(
                standardized_dataframe[column_name], errors="coerce"
            )

    # --- Drop rows without core variables ---
    cleaned_dataframe = standardized_dataframe.dropna(subset=["document_count", "total_citations"])

    # --- Save ---
    cleaned_dataframe.to_csv(CLEAN_OUTPUT_PATH, index=False)
    print(f"Cleaned file saved to: {CLEAN_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
