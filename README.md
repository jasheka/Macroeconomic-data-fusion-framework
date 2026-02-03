# Mixed-Frequency Macroeconomic Forecasting Framework

This project implements a mixed-frequency macroeconomic data analysis and forecasting framework using publicly available data from the FRED (Federal Reserve Economic Data) repository. The framework integrates MIDAS regression for mixed-frequency data alignment and LSTM networks for temporal feature extraction to support macroeconomic forecasting and policy-oriented analysis.

## Dataset
- **Source:** Federal Reserve Economic Data (FRED)
- **Link:** https://fred.stlouisfed.org/
- **Description:** The dataset includes macroeconomic indicators such as GDP, inflation rates, unemployment, interest rates, and other financial and economic time-series data at different frequencies (daily, monthly, quarterly).

## Methodology Overview
The proposed workflow consists of the following steps:
1. **Data Collection:** Retrieve high-frequency and low-frequency macroeconomic indicators from FRED.
2. **Data Preprocessing:** Clean, normalize, and handle missing values in the dataset.
3. **Mixed-Frequency Integration:** Apply MIDAS regression to align high-frequency and low-frequency data.
4. **Feature Extraction:** Use LSTM networks to capture short-term and long-term temporal patterns.
5. **Policy-Oriented Modeling:** Apply econometric models for forecasting and decision support.
6. **Evaluation:** Assess performance using MSE, MAE, and R² metrics.


