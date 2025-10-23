
# 📊 Bibliometric Regression Analysis of Library Marketing Impact (2010–2025)

This project performs a **bibliometric linear regression analysis** to explore the relationship between the **number of indexed research documents** and the **total citation impact** of authors publishing in the field of **library marketing**.

The analysis relies on bibliometric data extracted from **Web of Science (WoS)** and processed through **InCites (Clarivate Analytics)**.

<p align="center">
  <img src="https://github.com/sblando/bibliometrics-sql-regression/blob/45dd4342e9e1e9fc5c0359563569dc7f90fdb9fd/outputs/plots/regression_line.png?raw=true" 
       alt="Regression Line Plot" 
       width="600">
</p>

---

## 📥 Data Source

The dataset was retrieved from the **Web of Science Core Collection** and enriched with metrics using **InCites** (Clarivate Analytics). The exported CSV file includes the following variables:

---

## 🔍 Boolean Search Strategy

### Boolean query used to retrieve data:
 
> ("library marketing" OR "marketing in libraries" OR "library promotion") AND (impact OR evaluation OR   effectiveness OR engagement) NOT (editorial OR "book review" OR commentary)

###  Web of Science TS field equivalent:

> (((TS = "library marketing") OR (TS = "marketing in libraries")) OR (TS = "library promotion")) AND((((TS = impact) OR (TS = evaluation)) OR (TS = effectiveness)) OR (TS = engagement)) NOT((((TS = editorial) OR (TS = "book review")) OR (TS = commentary)))

## How to Run the Project

```bash
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows
```

```bash
# 2. Install dependencies
pip install -r requirements.txt
```

```bash
# 3. Clean raw dataset
python src/clean_data.py
```

```bash
# 4. Run regression analysis
python src/run_regression.py
```

---
