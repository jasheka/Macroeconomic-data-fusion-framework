# =========================
# STEP 0: FIX RANDOMNESS
# =========================
import os, random
import numpy as np
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# STEP 1: IMPORTS
# =========================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from statsmodels.tsa.api import VAR

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
# STEP 2: LOAD FRED DATA
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

# =========================
# STEP 3: MIDAS-STYLE ALIGNMENT
# =========================
data = data.resample("ME").mean()
data = data.dropna()
data = data.loc["2000-01-01":"2024-12-31"]

scaler = MinMaxScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns,
    index=data.index
)

# =========================
# STEP 4: LSTM FEATURE EXTRACTION (GDP)
# =========================
features = ["CPI", "UNRATE", "FEDFUNDS", "INDPRO", "M2SL", "DGS10", "RSAFS"]
target = "GDP"

def create_sequences(df, window=12):
    X, y = [], []
    for i in range(len(df) - window):
        X.append(df[features].iloc[i:i+window].values)
        y.append(df[target].iloc[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential([
    LSTM(
        128,
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ),
    LSTM(64),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=50,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=4,
    callbacks=[early_stop],
    verbose=1
)

y_pred_gdp = model.predict(X_test).flatten()

# =========================
# STEP 5: POLICY OPTIMIZATION (VAR MODEL)
# =========================
var_data = data_scaled.copy()

var_model = VAR(var_data)
var_results = var_model.fit(maxlags=6, ic="aic")

# Fitted values (policy-adjusted equilibrium)
var_fitted = var_results.fittedvalues

# Align for plotting
var_fitted = var_fitted.loc[data_scaled.index.intersection(var_fitted.index)]

# =========================
# STEP 6: EVALUATION (GDP)
# =========================
mse = mean_squared_error(y_test, y_pred_gdp)
mae = mean_absolute_error(y_test, y_pred_gdp)
r2  = r2_score(y_test, y_pred_gdp)

print("\n📊 FINAL GDP MODEL PERFORMANCE")
print("MSE :", mse)
print("MAE :", mae)
print("R²  :", r2)

# # =========================
# # STEP 7: GRAPHS
# # =========================
# # ---- 1️⃣ GDP Actual vs Predicted (LSTM)
# plt.figure(figsize=(10,7))
# plt.plot(range(len(y_test)), y_test, label="Actual GDP", color="blue", linewidth=2)
# plt.plot(range(len(y_pred_gdp)), y_pred_gdp, label="Predicted GDP", linestyle="--", color="orange", linewidth=2)
# plt.title("LSTM Prediction: Actual vs Predicted GDP", font)
# plt.xlabel("Test Sample Index", font)
# plt.ylabel("Scaled GDP Value", font)
# plt.legend(fontsize=17)
# plt.tight_layout()
# plt.savefig("graphs/LSTM_Prediction.png",dpi=1000)
# plt.show()
#
#
# # ---- 2️⃣ Regression Fit (GDP)
# plt.figure(figsize=(10,7))
# plt.scatter(y_test, y_pred_gdp, alpha=0.7, color="purple", label="Predicted vs Actual")
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()], linestyle="--", color="black", label="Perfect Fit")
# plt.title(f"GDP Regression Fit (R² = {r2:.3f})", font)
# plt.xlabel("Actual GDP", font)
# plt.ylabel("Predicted GDP", font)
# plt.legend(fontsize=17)
# plt.tight_layout()
# plt.savefig("graphs/GDP_Prediction.png",dpi=1000)
# plt.show()
#
# # ---- 3️⃣ Residuals (GDP)
# residuals = y_test - y_pred_gdp
# plt.figure(figsize=(10,7))
# plt.plot(residuals, label="Residuals", color="green", linewidth=2)
# plt.axhline(0, linestyle="--", color="red", linewidth=2)
# plt.title("GDP Residual Error Plot", font)
# plt.xlabel("Test Sample Index", font)
# plt.ylabel("Residual Value", font)
# plt.legend(fontsize=17)
# plt.tight_layout()
# plt.savefig("graphs/GDP_Prediction_Error.png",dpi=1000)
# plt.show()
#
# # ---- Training Curve
# plt.figure(figsize=(10,7))
# plt.plot(history.history["loss"], label="Training Loss",color="green",linewidth=1.5)
# plt.plot(history.history["val_loss"], label="Validation Loss",color="orange")
# plt.title("LSTM Training Curve", font)
# plt.xlabel("Epochs",font)
# plt.ylabel("Values",font)
# plt.legend(fontsize=16)
# plt.tight_layout()
# plt.savefig("graphs/LSTM_Training_Curve.png",dpi=1000)
# plt.show()
#
# # =========================
# # STEP 8: ACTUAL vs PREDICTED FOR ALL INDICATORS (VAR)
# # # =========================
# for col in var_fitted.columns:
#     plt.figure(figsize=(10,7))
#     plt.plot(data_scaled[col], label=f"Actual {col}",color="#574964",linewidth=2.5)
#     plt.plot(var_fitted[col], linestyle="--", label=f"Predicted {col}",linewidth=2,color="#980404")
#     plt.xlabel("Year",font)
#     plt.ylabel("Values",font)
#     plt.title(f"{col}",font)
#     plt.legend(fontsize=16)
#     plt.tight_layout()
#     plt.savefig(f"graphs/{col}.png",dpi=1000)
#     plt.show()

# Map FRED column codes to full descriptive names
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
# for col in var_fitted.columns:
#     plt.figure(figsize=(10,7))
#     plt.plot(data_scaled[col], label=f"Actual {col}", color="#001BB7", linewidth=2)
#     plt.plot(var_fitted[col], linestyle="--", label=f"Predicted {col}", linewidth=2.5, color="red")
#     plt.xlabel("Year", font)
#     plt.ylabel("Scaled Value", font)
#     plt.title(f"{fred_titles.get(col, col)}: Actual vs Predicted", font)
#     plt.legend(fontsize=17)
#     plt.tight_layout()
#     plt.savefig(f"graphs/{col}.png", dpi=1000)
#     plt.show()
#
