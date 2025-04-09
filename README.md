# Forestry Project: Forecasting Timber Prices in Finland Using SARIMA 
## Project Goal
This project analyzes and forecasts monthly timber prices for different tree species in Finland using SARIMA techniques. 
## Tools and Skills
- Python (pandas, numpy, matplotlib, seaborn, statsmodels)
- Time series modeling (SARIMA)
- Data wrangling and visualization
## Analysis workflow
1. Data cleaning
   - Extract data from [LUKE](https://statdb.luke.fi/PxWeb/pxweb/en/LUKE/LUKE__04%20Metsa__04%20Talous__02%20Teollisuuspuun%20kauppa__02%20Kuukausitilastot/01a_Kantohinnat_kk.px/?rxid=dc711a9e-de6d-454b-82c2-74ff79a3a5e0)
   - Handle missing values and removed irrelevant columns
2. Exploratory Data Analysis (EDA)
   - Visualize monthly price trends for pine, spruce, and birch logs
   - Conduct seasonal decomposition to inspect trend and seasonality patterns
3. Stationary and transform datasets
   - Apply ADF and KPSS tests to confrim stationarity
   - Go through log transformation and differencing (first or second order)
4. ACF and PACF visualization
   - Create autocorrelation (ACF) and partial autocorrelation (PACF) plots to support SARIMA order selection
5. Forecasting with SARIMA
   - Build SARIMA models using 'pmdarima.auto_arima' with different seasonal cycles
   - Forecaste next 12 months' timber prices for each tree species
   - Visualize historical vs. predicte values with confidence intervals
6. Model evaluation
   - Evaluate model performance with metrices: MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), SMAPE (Symmetric Mean Absolute Percentage Error), and MASE (Mean Absolute Scaled Error)
   - Perform time series cross-validation (3 folds) to assess model robustness
7. Visualizations
   - Time series line charts by species
   - Seasonal decomposition plots
   - ACF / PACF plots
   - SARIMA forecasts with confidence bands
   - Cross-validation RMSE comparison chart
