import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import math
from tabulate import tabulate
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read "monthlystumpage_standingsales" excel file
standing_sales = pd.read_excel("monthlystumpage_standingsales.xlsx", skiprows=2)

##----------------- CLEAN DATASETS----------------------------------##
# Conduct "Missing" values
standing_sales.isnull().sum() # no missing values
standing_sales.dtypes

# Convert "Time" column to ensure time-series datasets
standing_sales["Time"] = pd.to_datetime(standing_sales["Time"], format="%Y/%m")

##----------------- EDA analysis for standing sales---------------------## 
# Check summary statistics
statistics_standing_sales = standing_sales.describe()
print(tabulate(statistics_standing_sales, headers='keys', tablefmt='github'))

# Time series visualization
plt.figure(figsize=(10, 5))
sns.lineplot(data=standing_sales, x = "Time", y = "Pine logs", label="Pine logs")
sns.lineplot(data=standing_sales, x = "Time", y = "Spruce logs", label="Spruce logs")
sns.lineplot(data=standing_sales, x = "Time", y = "Birch logs", label="Birch logs")
plt.title("Standing sales timber prices over time by tree species")
plt.xlabel("Time")
plt.ylabel("Price (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Seasonal Decomposition
# Create decomposition function
def decompose_and_plot(series, time_index, title_prefix):
    series.index = time_index
    result = seasonal_decompose(series, model='additive', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    components = ['Observed', 'Trend', 'Seasonal', 'Residuals']
    data = [result.observed, result.trend, result.seasonal, result.resid]

    for ax, comp, d in zip(axes, components, data):
        if comp == 'Residuals':
            ax.scatter(d.index, d, color='steelblue', s=10)  
        else:
            ax.plot(d, color='steelblue')  
        ax.set_ylabel(comp, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title("")

    axes[-1].set_xlabel("Time", fontsize=11)

    plt.suptitle(f"Seasonal Decomposition of {title_prefix}", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

for species in ["Pine logs", "Spruce logs", "Birch logs"]:
    print(f"Decomposition for {species}")
    decompose_and_plot(standing_sales[species], standing_sales["Time"], species)

##-------------------------- Test stationary---------------------------##
stationarity_results = []

# ADF and KPSS test 
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

stationarity_df = pd.DataFrame(stationarity_results)
print(tabulate(stationarity_df, headers='keys', tablefmt='github'))

# Apply log transformation and first differencing
log_diff_values = pd.DataFrame()

for log_type in ["Pine logs", "Spruce logs", "Birch logs"]:
    log_series = np.log(standing_sales[log_type])
    log_diff_values[log_type] = log_series.diff()

log_diff_values = log_diff_values.dropna()

log_diff_values["Time"] = standing_sales["Time"].iloc[1:].values

log_diff_first = log_diff_values.reset_index(drop=True)

# Stationarity tests on first log-differenced data
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
print(tabulate(stationarity_result_log_first, headers='keys', tablefmt='github'))

log_diff = log_diff_first 
log_diff.set_index("Time", inplace=True)

## ------------------------------ ACF and PACF-------------------------##
# ACF and PACF 
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
log_types = ["Pine logs", "Spruce logs", "Birch logs"]
labels = ['a', 'b', 'c', 'd', 'e', 'f']

for i, col in enumerate(log_types):
    plot_acf(log_diff[col], ax=axes[i][0], lags=36, title=f"({labels[i*2]}) ACF - {col}")
    plot_pacf(log_diff[col], ax=axes[i][1], lags=36, title=f"({labels[i*2+1]}) PACF - {col}")

plt.tight_layout()

# Comments: Based on ACF and PACF figures to choose (p.d.q)(P, D, Q, s).
# Pine: p=1,5,7; d=1, q =1,2; P=1,2
# Spruce: p=1,7; d=1, q=1,2,7; P=2,3
# Birch: p=0,1; d=1, q=0; P=0,1
# D=1, s=12, Q=1

## -------------------------- SARIMA mode forecast ---------------------##
warnings.filterwarnings("ignore")

# Pine logs: p=1,5,7; d=1, q =1,2; P=1,2; D=1, s=12, Q=1
# Define (p.d.q)(P, D, Q, s) values
p_pine = [1,5,7]
d_pine = 1
q_pine = [1,2]
P_pine = [1,2]
D= 1
Q =1
s =12

results_pine = []

d= d_pine
for p in p_pine:
    for q in q_pine:
        for P in P_pine:
            try:
                model = SARIMAX(log_diff["Pine logs"],
                                order =(p,d,q),
                                seasonal_order = (P,D,Q,s),
                                enforce_stationarity = False,
                                enforce_invertibility =False)
                fit = model.fit(disp=False)
                results_pine.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": fit.aic,
                                     "BIC": fit.bic})
            except Exception as e:
                results_pine.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": np.nan,
                                     "BIC": np.nan,
                                     "Error": str(e)})
                
results_pine_df =pd.DataFrame(results_pine)
print(tabulate(results_pine_df, headers="keys", tablefmt="github"))

# Find the best model based on AIC and BIC (min values)
best_aic_pine = results_pine_df.loc[results_pine_df["AIC"].idxmin()]
best_bic_pine = results_pine_df.loc[results_pine_df["BIC"].idxmin()]

print("Best model for Pine logs by AIC:")
print(best_aic_pine)
print("Best model for Pine logs by BIC:")
print(best_bic_pine)

# Spruce logs: p=1,7; d=1, q=1,2,7; P=2,3; D=1, s=12, Q=1
# Define (p.d.q)(P, D, Q, s) values
p_spruce = [1,7]
d_spruce = 1
q_spruce = [1,2,7]
P_spruce = [2,3]
D= 1
Q =1
s =12

results_spruce = []

d=d_spruce
for p in p_spruce:
    for q in q_spruce:
        for P in P_spruce:
            try:
                model = SARIMAX(log_diff["Spruce logs"],
                                order =(p,d,q),
                                seasonal_order = (P,D,Q,s),
                                enforce_stationarity = False,
                                enforce_invertibility =False)
                fit = model.fit(disp=False)
                results_spruce.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": fit.aic,
                                     "BIC": fit.bic})
            except Exception as e:
                results_spruce.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": np.nan,
                                     "BIC": np.nan,
                                     "Error": str(e)})
results_spruce_df =pd.DataFrame(results_spruce)
print(tabulate(results_spruce_df, headers="keys", tablefmt="github"))

# Find the best model based on AIC and BIC (min values)
best_aic_spruce = results_spruce_df.loc[results_spruce_df["AIC"].idxmin()]
best_bic_spruce = results_spruce_df.loc[results_spruce_df["BIC"].idxmin()]

print("Best model for Spruce by AIC:")
print(best_aic_spruce)
print("Best model for Spruce by BIC:")
print(best_bic_spruce)

# Birch logs: Birch: p=0,1; d=1, q=0; P=0,1; D=1, s=12, Q=1
# Define (p.d.q)(P, D, Q, s) values
p_birch = [0,1]
d_birch = 1
q_birch = [0]
P_birch = [0,1]
D= 1
Q =1
s =12

results_birch = []

d= d_birch
for p in p_birch:
    for q in q_birch:
        for P in P_birch:
            try:
                model = SARIMAX(log_diff["Birch logs"],
                                order =(p,d,q),
                                seasonal_order = (P,D,Q,s),
                                enforce_stationarity = False,
                                enforce_invertibility =False)
                fit = model.fit(disp=False)
                results_birch.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": fit.aic,
                                     "BIC": fit.bic})
            except Exception as e:
                results_birch.append({"order": (p,d,q),
                                     "seasonal_order": (P,D,Q,s),
                                     "AIC": np.nan,
                                     "BIC": np.nan,
                                     "Error": str(e)})
results_birch_df =pd.DataFrame(results_birch)
print(tabulate(results_birch_df, headers="keys", tablefmt="github"))

# Find the best model based on AIC and BIC (min values)
best_aic_birch = results_birch_df.loc[results_birch_df["AIC"].idxmin()]
best_bic_birch = results_birch_df.loc[results_birch_df["BIC"].idxmin()]

print("Best model for Birch by AIC:")
print(best_aic_birch)
print("Best model for Birch by BIC:")
print(best_bic_birch)

## --------------------- Evaluate each log with SARIMA order ------------------------------##
# MAE and RMSE
forecast_horizon = 12

# Define best model orders
orders = {"Pine logs": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
          "Spruce logs": {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
          "Birch logs": {"order": (1, 1, 0), "seasonal_order": (0, 1, 1, 12)}}

forecast_metrics = []

# Evaluate each log type
for log_type, params in orders.items():
    series = log_diff[log_type].dropna()
    model = SARIMAX(series, order=params["order"], seasonal_order=params["seasonal_order"],
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=forecast_horizon)

    # Get true values for comparison
    actual = series[-forecast_horizon:]
    forecast = forecast[:len(actual)]

    # Calculate error metrics
    mae = mean_absolute_error(actual, forecast)
    rmse = math.sqrt(mean_squared_error(actual, forecast))

    forecast_metrics.append({"Tree Species": log_type,
                             "MAE": mae,"RMSE": rmse})

forecast_metrics_df = pd.DataFrame(forecast_metrics)
print(tabulate(forecast_metrics_df, headers="keys", tablefmt="github"))

# Cross validation with 3-month horizons
# Rolling forecast origin
def rolling_cv_rmse(series, order, seasonal_order, forecast_horizon=3, folds=3):
    rmses = []
    n = len(series)
    split_point = n - forecast_horizon * folds

    for i in range(folds):
        train_end = split_point + i * forecast_horizon
        train = series.iloc[:train_end]
        test = series.iloc[train_end:train_end + forecast_horizon]

        try:
            model = SARIMAX(train,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            fit = model.fit(disp=False)
            forecast = fit.forecast(steps=forecast_horizon)
            rmse = mean_squared_error(test, forecast, squared=False)
            rmses.append(rmse)
        except Exception as e:
            print(f"Error in fold {i+1}: {e}")
            rmses.append(np.nan)
    
    return rmses

pine_rmse_cv = rolling_cv_rmse(series=log_diff["Pine logs"],
                               order=(1, 1, 1),
                               seasonal_order=(1, 1, 1, 12),
                               forecast_horizon=3,
                               folds=3)
print("Pine logs CV RMSE:", pine_rmse_cv)

spruce_rmse_cv = rolling_cv_rmse(series=log_diff["Spruce logs"],
                               order=(1, 1, 1),
                               seasonal_order=(1, 1, 1, 12),
                               forecast_horizon=3,
                               folds=3)
print("Spruce logs CV RMSE:", spruce_rmse_cv)

birch_rmse_cv = rolling_cv_rmse(series=log_diff["Birch logs"],
                               order=(1, 1, 0),
                               seasonal_order=(0, 1, 1, 12),
                               forecast_horizon=3,
                               folds=3)
print("Birch logs CV RMSE:", birch_rmse_cv)

# Merge CV results
cv_results = {"Tree Species": ["Pine logs", "Spruce logs", "Birch logs"],
              "CV RMSE Fold 1": [pine_rmse_cv[0], spruce_rmse_cv[0], birch_rmse_cv[0]],
              "CV RMSE Fold 2": [pine_rmse_cv[1], spruce_rmse_cv[1], birch_rmse_cv[1]],
              "CV RMSE Fold 3": [pine_rmse_cv[2], spruce_rmse_cv[2], birch_rmse_cv[2]],
              "CV RMSE Mean": [np.mean(pine_rmse_cv), 
                               np.mean(spruce_rmse_cv), 
                               np.mean(birch_rmse_cv)],
              "CV RMSE Std": [np.std(pine_rmse_cv), 
                               np.std(spruce_rmse_cv), 
                               np.std(birch_rmse_cv)]}

cv_results_df = pd.DataFrame(cv_results)
print(tabulate(cv_results_df, headers="keys", tablefmt="github"))

species = cv_results_df["Tree Species"]
means = cv_results_df["CV RMSE Mean"]
stds = cv_results_df["CV RMSE Std"]

plt.figure(figsize=(8,5))
bars = plt.bar(species, means, yerr=stds, capsize=8, label="Mean RMSE")
plt.ylabel("CV RMSE")
plt.title("Cross-Validation RMSE with Standard Deviation")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(["Mean RMSE ± Std (error bar)"], loc="upper left")

for bar, mean in zip(bars, means):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{mean:.4f}", ha='center', va='bottom')
plt.tight_layout()

## -------------------------- Forecasting -------------------------##
# Forecasting 
confidence_level = 0.95

# Pine logs
model_pine = SARIMAX(log_diff["Pine logs"],
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
fit_pine = model_pine.fit(disp=False)

forecast_result_pine = fit_pine.get_forecast(steps=forecast_horizon)
forecast_pine = forecast_result_pine.predicted_mean
conf_int_pine = forecast_result_pine.conf_int(alpha=1-confidence_level)

# Fix index: forecast time should be monthly steps
last_date = log_diff.index[-1]
forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
forecast_pine.index = forecast_index
conf_int_pine.index = forecast_index

# Plot with observed data
plt.figure(figsize=(10, 5))
plt.plot(log_diff["Pine logs"], label="Historical data", color='steelblue')
plt.plot(forecast_pine, label="Forecast", color='darkorange')
plt.fill_between(conf_int_pine.index, conf_int_pine.iloc[:, 0], conf_int_pine.iloc[:, 1],
                 color='orange', alpha=0.3, label="95% Confidence")
plt.title("Pine Logs Forecast (SARIMA)")
plt.xlabel("Time")
plt.ylabel("Differenced log price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Spruce logs
model_spruce = SARIMAX(log_diff["Spruce logs"],
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
fit_spruce = model_spruce.fit(disp=False)

forecast_result_spruce = fit_spruce.get_forecast(steps=forecast_horizon)
forecast_spruce = forecast_result_spruce.predicted_mean
conf_int_spruce = forecast_result_spruce.conf_int(alpha=1-confidence_level)

# Fix index: forecast time should be monthly steps
forecast_spruce.index = forecast_index
conf_int_spruce.index = forecast_index

# Plot with observed data
plt.figure(figsize=(10, 5))
plt.plot(log_diff["Spruce logs"], label="Historical data", color='steelblue')
plt.plot(forecast_spruce, label="Forecast", color='darkorange')
plt.fill_between(conf_int_spruce.index, conf_int_spruce.iloc[:, 0], conf_int_spruce.iloc[:, 1],
                 color='orange', alpha=0.3, label="95% Confidence")
plt.title("Spruce Logs Forecast (SARIMA)")
plt.xlabel("Time")
plt.ylabel("Differenced log price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Birch logs
model_birch = SARIMAX(log_diff["Birch logs"],
                     order=(1, 1, 0),
                     seasonal_order=(0, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)
fit_birch = model_birch.fit(disp=False)

forecast_result_birch = fit_birch.get_forecast(steps=forecast_horizon)
forecast_birch = forecast_result_birch.predicted_mean
conf_int_birch = forecast_result_birch.conf_int(alpha=1-confidence_level)

# Fix index: forecast time should be monthly steps
forecast_birch.index = forecast_index
conf_int_birch.index = forecast_index

# Plot with observed data
plt.figure(figsize=(10, 5))
plt.plot(log_diff["Birch logs"], label="Historical data", color='steelblue')
plt.plot(forecast_birch, label="Forecast", color='darkorange')
plt.fill_between(conf_int_birch.index, conf_int_birch.iloc[:, 0], conf_int_birch.iloc[:, 1],
                 color='orange', alpha=0.3, label="95% Confidence")
plt.title("Birch Logs Forecast (SARIMA)")
plt.xlabel("Time")
plt.ylabel("Differenced log price")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

