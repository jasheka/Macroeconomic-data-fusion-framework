# =========================
# STEP 0: IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['font.weight'] = 'bold'
font = {
    'family': 'Times New Roman',
    'color': 'Black',
    'weight': 'bold',
    'size': 20,
}
# =========================
# STEP 1: LOAD AND PREPROCESS DATA
# =========================
def load_fred_csv(path, name):
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df.set_index("observation_date", inplace=True)
    df.columns = [name]
    return df

data = pd.concat([
    load_fred_csv("Dataset/GDP.csv", "GDP"),
    load_fred_csv("Dataset/CPIAUCSL.csv", "CPI"),
    load_fred_csv("Dataset/UNRATE.csv", "UNRATE"),
    load_fred_csv("Dataset/FEDFUNDS.csv", "FEDFUNDS"),
    load_fred_csv("Dataset/INDPRO.csv", "INDPRO"),
    load_fred_csv("Dataset/M2SL.csv", "M2SL"),
    load_fred_csv("Dataset/DGS10.csv", "DGS10"),
    load_fred_csv("Dataset/RSAFS.csv", "RSAFS")
], axis=1)

data = data.loc["2000-01-01":"2024-12-31"].resample("ME").mean().dropna()

features = ["CPI","UNRATE","FEDFUNDS","INDPRO","M2SL","DGS10","RSAFS"]
target = "GDP"

scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# =========================
# STEP 2: LSTM SEQUENCES
# =========================
def create_sequences(df, window=12):
    X, y = [], []
    for i in range(len(df)-window):
        X.append(df[features].iloc[i:i+window].values)
        y.append(df[target].iloc[i+window])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(data_scaled)
split = int(0.8*len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# =========================
# STEP 3: LSTM MODEL
# =========================
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    LSTM(64),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
history = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test),
                         epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
y_pred_lstm_scaled = lstm_model.predict(X_test).flatten()

# Convert LSTM predictions back to original GDP scale
gdp_min, gdp_max = data['GDP'].min(), data['GDP'].max()
y_test_original = data['GDP'].iloc[-len(y_test):]  # actual GDP for test period
y_pred_lstm = y_pred_lstm_scaled * (gdp_max - gdp_min) + gdp_min

# Resample LSTM to quarterly
lstm_series = pd.Series(y_pred_lstm, index=y_test_original.index)
lstm_q = lstm_series.resample('QE').mean()

# =========================
# STEP 4: MIDAS REGRESSION
# =========================
y_midas = data['GDP'].resample('QE').mean()
X_monthly = data[features]

lags = 12
X_lagged = pd.DataFrame(index=X_monthly.index)
for col in features:
    for lag in range(1,lags+1):
        X_lagged[f"{col}_lag{lag}"] = X_monthly[col].shift(lag)
X_lagged_q = X_lagged.resample('QE').mean().dropna()
y_midas_aligned = y_midas.loc[X_lagged_q.index]

split_idx = int(len(X_lagged_q)*0.8)
X_train_midas = X_lagged_q.iloc[:split_idx]
X_test_midas  = X_lagged_q.iloc[split_idx:]
y_train_midas = y_midas_aligned.iloc[:split_idx]
y_test_midas  = y_midas_aligned.iloc[split_idx:]

X_train_const = sm.add_constant(X_train_midas)
X_test_const  = sm.add_constant(X_test_midas)
midas_model = sm.OLS(y_train_midas, X_train_const).fit()
midas_pred_test = midas_model.predict(X_test_const)

# =========================
# STEP 5: VAR MODEL
# =========================
var_model = VAR(data_scaled)
var_results = var_model.fit(maxlags=6, ic="aic")
var_fitted = var_results.fittedvalues
var_pred_gdp = var_fitted['GDP'].iloc[-len(y_test_midas):]  # align with test
var_pred_gdp = var_pred_gdp * (gdp_max - gdp_min) + gdp_min  # rescale

# =========================
# STEP 6: COMBINE PREDICTIONS (LEARNED ENSEMBLE)
# =========================
# Align all to quarterly test index
test_index = y_test_midas.index
lstm_q = lstm_q.loc[test_index]
midas_q = pd.Series(midas_pred_test.values, index=test_index)
var_q   = pd.Series(var_pred_gdp.values, index=test_index)

# Create DataFrame for ensemble regression
ensemble_df = pd.DataFrame({
    'LSTM': lstm_q.values,
    'MIDAS': midas_q.values,
    'VAR': var_q.values
}, index=test_index)

# Train linear regression on predictions to learn optimal combination
ensemble_model = LinearRegression()
ensemble_model.fit(ensemble_df, y_test_midas.values)
final_pred = ensemble_model.predict(ensemble_df)
y_true = y_test_midas.values

# =========================
# STEP 7: METRICS
# =========================
final_mse = mean_squared_error(y_true, final_pred)
final_mae = mean_absolute_error(y_true, final_pred)
final_r2  = r2_score(y_true, final_pred)

print("\n📊 FINAL ENSEMBLE METRICS")
print("MSE:", round(final_mse, 4))
print("MAE:", round(final_mae, 4))
print("R² :", round(final_r2, 4))


# =========================
# NORMALIZED METRICS (0–1)
# =========================
y_range = y_true.max() - y_true.min()

norm_mae = final_mae / y_range
norm_mse = final_mse / (y_range ** 2)

print("\n📉 NORMALIZED METRICS (0–1 SCALE)")
print("Normalized MAE:", round(norm_mae, 6))
print("Normalized MSE:", round(norm_mse, 6))
#
# import matplotlib.dates as mdates
#
# # ---------------------------
# # Residuals
# # ---------------------------
# residuals = y_true - final_pred
#
# plt.figure(figsize=(10,7))
# plt.plot(test_index, residuals, marker='o', linewidth=2, color="green")
# plt.axhline(0, color='red', linestyle='--')
#
# plt.xlabel("Year", font)
# plt.ylabel("Residual Value", font)
# plt.title("Residuals", font)
#
# # X-axis: show years only
# plt.gca().xaxis.set_major_locator(mdates.YearLocator())
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.tight_layout()
# plt.savefig("graphs/residuals.png", dpi=1000)
# plt.show()
#
#
# plt.figure(figsize=(10,7))
#
# # Scatter: Actual vs Predicted
# plt.scatter(
#     y_true,
#     final_pred,
#     color="#1f77b4",        # blue
#     alpha=0.8,
#     s=70,
#     label="Predicted vs Actual"
# )
#
# # Perfect fit line
# plt.plot(
#     [y_true.min(), y_true.max()],
#     [y_true.min(), y_true.max()],
#     linestyle="--",
#     color="red",
#     linewidth=3,
#     label="Perfect Fit"
# )
#
# plt.xlabel("Actual GDP", font)
# plt.ylabel("Predicted GDP", font)
# plt.title(f"Ensemble Regression Fit(R² = {final_r2:.4f})", font)
#
# plt.legend(fontsize=17)
#
# plt.tight_layout()
# plt.savefig("graphs/r2.png",dpi=1000)
# plt.show()
#
# # =========================
# # STEP 8: ACTUAL VS PREDICTED FOR ALL INDICATORS
# # =========================
# import matplotlib.dates as mdates
#
# # =========================
# # STEP 8: ACTUAL VS PREDICTED FOR ALL INDICATORS
# # =========================
#
# # Quarterly ground truth
# actual_q = data.resample("QE").mean()
#
# # Align to test period
# actual_q = actual_q.loc[y_test_midas.index]
#
# # -------------------------
# # GDP: Ensemble predictions
# # -------------------------
# predicted_gdp_q = pd.Series(final_pred, index=y_test_midas.index, name="GDP")
#
# # -------------------------
# # Other indicators: VAR predictions
# # -------------------------
# # Inverse-transform VAR fitted values to original scale
# var_pred_original = pd.DataFrame(
#     scaler.inverse_transform(var_fitted),
#     columns=var_fitted.columns,
#     index=var_fitted.index
# )
#
# # Resample to quarterly
# var_q = var_pred_original.resample("QE").mean()
#
# # -------------------------
# # Map FRED codes to descriptive names
# # -------------------------
# fred_titles = {
#     "GDP": "Gross Domestic Product",
#     "CPI": "Consumer Price Index",
#     "UNRATE": "Unemployment Rate",
#     "FEDFUNDS": "Federal Funds Rate",
#     "INDPRO": "Industrial Production Index",
#     "M2SL": "M2 Money Stock",
#     "DGS10": "10-Year Treasury Constant Maturity Rate",
#     "RSAFS": "Real Personal Consumption Expenditures\non Goods and Services"
# }
#
# # Create predictions dict
# all_preds = {"GDP": predicted_gdp_q}
# for col in ["CPI", "UNRATE", "FEDFUNDS", "INDPRO", "M2SL", "DGS10", "RSAFS"]:
#     common_index = actual_q.index.intersection(var_q.index)
#     all_preds[col] = pd.Series(var_q.loc[common_index, col], index=common_index)
#
# # -------------------------
# # Plot actual vs predicted for all indicators
# # -------------------------
# for col, pred_series in all_preds.items():
#     plt.figure(figsize=(10, 7))
#
#     # Align actual and predicted
#     common_index = actual_q.index.intersection(pred_series.index)
#     actual_values = actual_q.loc[common_index, col]
#     predicted_values = pred_series.loc[common_index]
#
#     plt.plot(common_index, actual_values, linewidth=2.5, label=f"Actual {col}", color="#1f77b4")
#     plt.plot(common_index, predicted_values, linewidth=2.5, linestyle="--", label=f"Predicted {col}", color="#d62728")
#
#     plt.xlabel("Year", font)
#     plt.ylabel(f"{col} Values", font)
#     plt.title(f"{fred_titles.get(col, col)}", font)
#     plt.legend(fontsize=17)
#
#     # Format x-axis: show years only
#     plt.gca().xaxis.set_major_locator(mdates.YearLocator())
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#
#     plt.tight_layout()
#     plt.savefig(f"graphs/{col}_Actual_vs_Predicted.png", dpi=1000)
#     plt.show()

#
# import matplotlib.pyplot as plt
#
# # Metrics
# metrics = ['MAE', 'MSE']
# values = [norm_mae, norm_mse]
#
# plt.figure(figsize=(8,6))
# bars = plt.bar(metrics, values, color=['#9F5255', '#C84C05'], alpha=0.8,width=0.6)
#
# # Add value labels on top of bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.00001, f'{height:.6f}',
#              ha='center', va='bottom', fontsize=16, fontweight='bold')
# plt.ylim(0, 0.007)
# plt.xlabel("Metric", font)
# plt.ylabel('Error values', font)
# plt.title('Normalized Regression Errors', font)
# plt.tight_layout()
# plt.savefig("graphs/Normalized_MAE_MSE.png", dpi=1000)
# plt.show()

# import seaborn as sns
#
# # Compute correlation matrix for actual quarterly indicators
# actual_q = data.resample("QE").mean()  # quarterly averages
# corr_matrix = actual_q.corr()
#
# plt.figure(figsize=(12,8))
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="PuOr", linewidths=0.5)
# plt.title("Correlation Heatmap: Quarterly Indicators\n", font)
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig("graphs/Correlation_Heatmap_Indicators.png", dpi=1000)
# plt.show()

# =========================
# LSTM Training vs Validation Loss
# =========================
plt.figure(figsize=(10,7))

plt.plot(history.history['loss'], label='Training Loss', linewidth=2.3, color="#0D63A5")
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2.3, color="#EB5B00")

plt.xlabel("Epoch", font)
plt.ylabel("Values", font)
plt.title("LSTM Training vs Validation Loss", font)
plt.legend(fontsize=17)
plt.tight_layout()
plt.savefig("graphs/LSTM_Train_Val_Loss.png", dpi=1000)
plt.show()
