import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.tsa.stattools as tsa
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read stumpage prices file
stumpage_prices = pd.read_excel("monthlyStumpagePrices.xlsx", skiprows=2)

# Clean datasets
# Conversions to "Time" column and fillings missing values to 0
stumpage_prices["Time"] = pd.to_datetime(stumpage_prices["Time"], format="%Y/%m")
stumpage_prices.iloc[:, 0] = stumpage_prices.iloc[:, 0].ffill()
stumpage_prices = stumpage_prices.fillna(0)
stumpage_prices.iloc[:, -3:] = stumpage_prices.iloc[:, -3:].apply(pd.to_numeric, errors="coerce")

# Filter out "Region" column
stumpage_prices = stumpage_prices.drop(columns=["Region"])

# Filter "Standing sales, total" in the "Felling method" column
standing_sales = stumpage_prices[stumpage_prices["Felling method"] == "Standing sales, total"].reset_index(drop=True)
standing_sales = standing_sales.drop(columns=["Felling method"])

# EDA analysis for standing sales 
# Check summary statistics
statistics_standing_sales = standing_sales.describe()
print(statistics_standing_sales)

# Time series visualization
plt.figure(figsize = (12,5))
sns.lineplot(data=standing_sales, x = "Time", y = "Pine logs", label="Pine logs")
sns.lineplot(data=standing_sales, x = "Time", y = "Spruce logs", label="Spruce logs")
sns.lineplot(data=standing_sales, x = "Time", y = "Birch logs", label="Birch logs")
plt.title("Standing sales timber prices over time by tree species")
plt.xlabel("Time")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Seasonal Decomposition
# Create decomposition function
def decompose_and_plot(series, time_index, title_prefix):
    series.index = time_index  
    result = seasonal_decompose(series, model='additive', period=12)

    plt.figure(figsize=(10, 6))
    result.plot()
    plt.suptitle(f"Seasonal Decomposition of {title_prefix} Stumpage Prices", fontsize=14)
    plt.tight_layout()
    plt.show()

# Decomposition for each log
for species in ["Pine logs", "Spruce logs", "Birch logs"]:
    print(f"Decomposition for {species} ")
    decompose_and_plot(standing_sales[species], standing_sales["Time"], species)

# Test for stationary
# Prepare the test results storage
stationarity_results = []

# Peform ADF and KPSS test for each log type
for log_type in ["Pine logs", "Spruce logs", "Birch logs"]:
    series = standing_sales[log_type].dropna()
    adf_result = adfuller(series, autolag = "AIC") # ADF Test
    adf_p = adf_result[1]
    
    try:
        kpss_result = kpss(series, regression = "c", nlags="auto") # KPSS test
        kpss_p = kpss_result[1]
    except:
        kpss_p = "Failed"
    
    stationarity_results.append({
        "Log Type": log_type,
        "ADF p-value": adf_p,
        "KPSS p-value": kpss_p})

# Convert results to dataframe for display
stationarity_df = pd.DataFrame(stationarity_results)
print(stationarity_df)

# Apply log transformation and first differencing
log_diff_values = pd.DataFrame()

for log_type in ["Pine logs", "Spruce logs", "Birch logs"]:
    log_series = np.log(standing_sales[log_type])
    log_diff_values[log_type] = log_series.diff()

log_diff_values = log_diff_values.dropna()

log_diff_values["Time"] = standing_sales["Time"].iloc[1:].values

log_diff_first = log_diff_values.reset_index(drop=True)

# Re-run ADF and KPSS tests on the log-differenced data
stationarity_result_log_first = []

for log_type in ["Pine logs", "Spruce logs", "Birch logs"]:
    series_log = log_diff_first[log_type].dropna()
    # ADF test
    adf_result_log_first = adfuller(series_log, autolag="AIC")
    adf_p_log_first = adf_result_log_first[1]
    
    # KPSS test
    try:
        kpss_result_log_first = kpss(series_log, regression="c", nlags="auto")
        kpss_p_log_first = kpss_result_log_first[1]
    except:
        kpss_p_log_first = "Failed"
    
    stationarity_result_log_first.append({
        "Log Type": log_type,
        "ADF p-value": adf_p_log_first,
        "KPSS p-value": kpss_p_log_first})

stationarity_result_log_first = pd.DataFrame(stationarity_result_log_first)
print(stationarity_result_log_first)

# Only apply second differencing on log-transformed Birch logs
birch_log = np.log(standing_sales["Birch logs"])
birch_log_diff2 = birch_log.diff().diff().dropna()

# Run ADF and KPSS tests again on the 2nd differences Birch logs
adf_birch_2nd = adfuller(birch_log_diff2, autolag="AIC")[1]
try:
    kpss_birch_2nd = kpss(birch_log_diff2, regression="c", nlags="auto")
    kpss_birch_2nd_p = kpss_birch_2nd[1]
except:
    kpss_birch_2nd = "Failed"

# Print result
birch_result = pd.DataFrame([{
    "Log Type": "Birch logs (2nd diff)",
    "ADF p-value": adf_birch_2nd,
    "KPSS p-value": kpss_birch_2nd_p}])
print(birch_result)

# Create 2nd-order differenced birch log with Time
birch_log_diff2_df = pd.DataFrame({
    "Birch logs": birch_log_diff2.values,
    "Time": standing_sales["Time"].iloc[2:].values  # shift by 2 for 2nd diff
})

# Ensure Time column is datetime format
birch_log_diff2_df["Time"] = pd.to_datetime(birch_log_diff2_df["Time"])

# Cut Pine & Spruce to match Birch length
n = birch_log_diff2_df.shape[0]
log_diff_trimmed = log_diff_first[["Pine logs", "Spruce logs"]].iloc[-n:].reset_index(drop=True)

# Combine and align all logs with proper Time index
log_diff = pd.concat([
    log_diff_trimmed,
    birch_log_diff2_df[["Birch logs"]].reset_index(drop=True),
    birch_log_diff2_df[["Time"]].reset_index(drop=True)
], axis=1)

# Set Time as index (key step!)
log_diff.set_index("Time", inplace=True)

# ACF and PACF
# Create ACF and PACF plots for each log type
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
log_types = ["Pine logs", "Spruce logs", "Birch logs"]
labels = ['a', 'b', 'c', 'd', 'e', 'f']

for i, col in enumerate(log_types):
    plot_acf(log_diff[col], ax=axes[i][0], lags=36, title=f"({labels[i*2]}) ACF - {col}")
    plot_pacf(log_diff[col], ax=axes[i][1], lags=36, title=f"({labels[i*2+1]}) PACF - {col}")

plt.tight_layout()
plt.show()

#%%%% SARIMA mode forecast

#log_diff["Time"] = pd.to_datetime(log_diff["Time"])
#log_diff.set_index("Time", inplace=True)

# Forecast settings
forecast_horizon = 12
confidence_level = 0.95

# Save results for SARIMA mode
forecast_sarima ={}

for column in log_diff.columns:
    series_sarima = log_diff[column].dropna()
    
    best_model_sarima = None
    best_aic_sarima = np.inf
    best_result_sarima = {}
    
    # Seasonal values of (12,4,1)
    for m in [12, 4, 1]:
        try:
            model_sarima = auto_arima(
                series_sarima,
                seasonal = True,
                m = m,
                stepwise=True,
                suppress_warnings= True,
                error_action='ignore',
                trace = False)
            if model_sarima.aic() < best_aic_sarima:
                forecast, conf_int = model_sarima.predict(n_periods = forecast_horizon, return_conf_int=True, alpha=1 - confidence_level)
                best_model_sarima = model_sarima
                best_aic_sarima = model_sarima.aic()
                best_result_sarima = {
                    "SARIMA forecast": forecast,
                    "conf_int": conf_int,
                    "SARIMA mode": model_sarima}
        except Exception as e:
            continue
    # Save results
    forecast_sarima[column] = best_result_sarima
    
    # Plot forecast
    forecast_sarima_index = pd.date_range(start=series_sarima.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
    plt.figure(figsize=(10, 5))
    plt.plot(series_sarima.index, series_sarima, label="Historical")
    plt.plot(forecast_sarima_index, best_result_sarima["SARIMA forecast"], label="SARIMA Forecast", linestyle="--")
    plt.fill_between(forecast_sarima_index, best_result_sarima["conf_int"][:, 0], best_result_sarima["conf_int"][:, 1],
                     color="orange", alpha=0.3, label="95% CI")
    plt.title(f"SARIMA Forecast for {column}")
    plt.xlabel("Time")
    plt.ylabel("Differenced Log Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
    
# Define metric calculation functions
def calculate_forecast_sarima_accuracy(actual, predicted):
    mae_sarima = np.mean(np.abs(actual - predicted))
    rmse_sarima = np.sqrt(np.mean((actual - predicted) ** 2))
    smape_sarima = 100 * np.mean(2*np.abs(actual - predicted)/(np.abs(actual)+np.abs(predicted)))
    # For MASE, use naive seasonal difference (lag=12 if possible, else lag=1)
    if len(actual) >= 13:
        naive_diff_sarima = np.abs(actual[12:] - actual[:-12])
    else:
        naive_diff_sarima = np.abs(actual[1:] - actual[:-1])
    mase_denom_sarima = np.mean(naive_diff_sarima) if len(naive_diff_sarima) > 0 else np.nan
    mase_sarima = mae_sarima / mase_denom_sarima if mase_denom_sarima != 0 else np.nan
    return mae_sarima, rmse_sarima, smape_sarima, mase_sarima

# Store model summary and forecast accuracy
model_sarima_metrics = []

for column, result in forecast_sarima.items():
    model_sarima = result["SARIMA mode"]
    forecast = result["SARIMA forecast"]
    actual = log_diff[column].dropna()
    # Ensure we align actual and predicted for the last forecast_horizon periods (in-sample evaluation)
    actual_eval = actual[-forecast_horizon:]
    forecast_eval = model_sarima.predict_in_sample()[-forecast_horizon:]
    
    # Calculate metircs
    mae_sarima, rmse_sarima, smape_sarima, mase_sarima = calculate_forecast_sarima_accuracy(actual_eval, forecast_eval)
    
    model_sarima_metrics.append({
        "Log Type": column,
        "AIC": model_sarima.aic(),
        "BIC": model_sarima.bic(),
        "Model Order": model_sarima.order,
        "Seasonal Order": model_sarima.seasonal_order,
        "MAE": mae_sarima,
        "RMSE": rmse_sarima,
        "SMAPE": smape_sarima,
        "MASE": mase_sarima})
    
# Display results as DataFrame
metrics_model_sarima = pd.DataFrame(model_sarima_metrics)
print(metrics_model_sarima)

# Cross-validation with 3-month horizons
# Define SMAPE function 
def smape_cv(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Define SARIMA time-series cross-validation function 
def time_series_cv_sarima(series_cv_sarima, forecast_horizon_cv_sarima=3, n_folds=3):
    fold_size_sarima = forecast_horizon_cv_sarima
    total_len_sarima = len(series_cv_sarima)
    results_cv_sarima = []
    
    for i in range(n_folds):
        train_end_sarima = total_len_sarima - fold_size_sarima * (n_folds -i)
        train_sarima = series_cv_sarima[:train_end_sarima]
        test_sarima = series_cv_sarima[train_end_sarima:train_end_sarima + fold_size_sarima]
        
        try:
            model_cv_sarima = auto_arima(train_sarima, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
            forecast_cv_sarima = model_cv_sarima.predict(n_periods=forecast_horizon_cv_sarima)
            mae_cv_sarima = mean_absolute_error(test_sarima, forecast_cv_sarima)
            rmse_cv_sarima = np.sqrt(mean_squared_error(test_sarima, forecast_cv_sarima))
            smape_val_cv_sarima = smape_cv(test_sarima, forecast_cv_sarima)
            results_cv_sarima.append({"Fold": i + 1, "MAE": mae_cv_sarima, "RMSE": rmse_cv_sarima, "SMAPE": smape_val_cv_sarima})
        except:
            results_cv_sarima.append({"Fold": i + 1, "MAE": np.nan, "RMSE": np.nan, "SMAPE": np.nan})

    return pd.DataFrame(results_cv_sarima) 
            
# Run CV for all log types
cv_results_sarima_all = []
for column in log_diff.columns:
    series_cv_sarima = log_diff[column].dropna()
    cv_results_sarima = time_series_cv_sarima(series_cv_sarima)
    cv_results_sarima["Log Type"] = column
    cv_results_sarima_all.append(cv_results_sarima)
    
# Combine and display results
cv_sarima = pd.concat(cv_results_sarima_all, ignore_index = True)
print(cv_sarima)

# Plot cross validation anaylsis for SARIMA mode
sns.set(style = "whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(data=cv_sarima, x="Log Type", y="RMSE", hue="Fold", palette="viridis")

plt.title("Comparison Across Folds for Each Log Type (SARIMA Cross Validation)")
plt.ylabel("RMSE")
plt.xlabel("Log Type")
plt.legend(title="Fold")
plt.tight_layout()
plt.show()
    
