import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.decomposition import PCA
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_excel('merged_stations.xlsx')

# Display DataFrame dimensions
print("\n=== Excel File Dimensions ===")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Display all column names
print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "="*50 + "\n")

# Display initial DataFrame info to understand the structure
print("Initial dataframe info:")
print(f"Shape: {df.shape}")
print("First few columns:", df.columns[:10].tolist())

# Check for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns detected: {categorical_columns}")

# ENHANCEMENT 1: Better Data Cleaning
print("\n----- ENHANCED DATA CLEANING -----")
# Remove columns with too many missing values (using a threshold of 20%)
threshold = len(df) * 0.2
df = df.drop(columns=[col for col in df.columns if
                      "Unnamed" in col or
                      df[col].isnull().sum() > threshold])

# More robust pH conversion with error handling
if 'pH (units)' in df.columns:
    # First try converting to numeric
    df['pH (units)'] = pd.to_numeric(df['pH (units)'], errors='coerce')

    # Check for invalid pH values (outside range 0-14)
    invalid_ph = ((df['pH (units)'] < 0) | (df['pH (units)'] > 14)).sum()
    if invalid_ph > 0:
        print(f"Detected {invalid_ph} invalid pH values outside range 0-14")
        # Replace invalid values with median
        valid_ph = df.loc[(df['pH (units)'] >= 0) & (df['pH (units)'] <= 14), 'pH (units)']
        if not valid_ph.empty:
            median_ph = valid_ph.median()
            df.loc[(df['pH (units)'] < 0) | (df['pH (units)'] > 14), 'pH (units)'] = median_ph
            print(f"Replaced invalid pH values with median: {median_ph}")

# Only drop rows with missing target values
df = df.dropna(subset=['Chlorophyll-a (ug/L)'])

# Instead of one-hot encoding, create a numeric station ID
if 'Station' in df.columns:
    print("Creating numeric station ID...")
    # Map each station to a unique numeric ID
    station_map = {station: i for i, station in enumerate(df['Station'].unique())}
    df['station_id'] = df['Station'].map(station_map)
    print(f"Created station_id feature with {len(station_map)} unique values")

# ENHANCEMENT 2: Better missing value imputation - using median instead of mean (more robust to outliers)
print("\n----- ENHANCED MISSING VALUE HANDLING -----")
# First identify columns with missing values
cols_with_na = df.columns[df.isna().any()].tolist()
print(f"Columns with missing values: {cols_with_na}")

# For numeric columns, impute with median (more robust than mean for skewed data)
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isna().any():
        na_count = df[col].isna().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed {na_count} missing values in '{col}' with median: {median_val:.4f}")

# Drop all remaining object/string columns that can't be converted to numeric
print("\nDropping remaining non-numeric columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"  - Dropping: {col}")
    df = df.drop(columns=[col])

# ENHANCEMENT 3: Advanced Outlier Detection and Handling
print("\n----- ADVANCED OUTLIER DETECTION AND HANDLING -----")

# Import the necessary libraries
from sklearn.ensemble import IsolationForest


# Create a function to detect and handle outliers using Isolation Forest
def detect_outliers_isolation_forest(df, columns, contamination=0.05):
    """
    Detect outliers using Isolation Forest algorithm (multivariate approach)

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    columns : list
        List of column names to use for outlier detection
    contamination : float, default=0.05
        The expected proportion of outliers in the data

    Returns:
    --------
    outlier_mask : boolean array
        True for outliers, False for inliers
    """
    # Select numeric columns only
    X = df[columns].select_dtypes(include=np.number)

    # Fill NaN values with median to make the algorithm work
    X = X.fillna(X.median())

    # Initialize and fit the isolation forest model
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    # Fit and predict
    # -1 for outliers and 1 for inliers
    y_pred = model.fit_predict(X)

    # Convert to boolean mask (True for outliers)
    outlier_mask = y_pred == -1

    # Print number of detected outliers
    n_outliers = np.sum(outlier_mask)
    print(f"Isolation Forest detected {n_outliers} outliers in {len(columns)} variables")

    return outlier_mask


# Create function to apply winsorization at specific percentiles
def winsorize_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """
    Winsorize outliers in a column using percentile thresholds

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    column : str
        Column name to winsorize
    lower_percentile : float, default=0.01
        Lower percentile threshold (e.g., 0.01 for 1st percentile)
    upper_percentile : float, default=0.99
        Upper percentile threshold (e.g., 0.99 for 99th percentile)
    """
    if column not in df.columns:
        return 0

    # Get original values for comparison
    orig_values = df[column].copy()

    # Calculate percentile thresholds
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)

    # Apply thresholds
    df.loc[df[column] < lower_bound, column] = lower_bound
    df.loc[df[column] > upper_bound, column] = upper_bound

    # Count changed values
    n_changed = (orig_values != df[column]).sum()

    if n_changed > 0:
        print(
            f"Winsorized {n_changed} values in '{column}' using {lower_percentile:.2f}-{upper_percentile:.2f} percentiles")

    return n_changed


# Minimal preprocessing function - useful for tree-based models that are robust to outliers
def minimal_preprocessing(df, columns):
    """
    Apply minimal preprocessing suitable for tree-based models like GBR
    - Just remove infinite values
    - Replace NaNs with median
    """
    for col in columns:
        if col in df.columns:
            # Replace inf/-inf with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Count NaNs
            na_count = df[col].isna().sum()

            # Replace NaNs with median
            if na_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Replaced {na_count} NaN/infinite values in '{col}' with median: {median_val:.4f}")

    return df



# Replace the current IQR-based outlier handling with this improved approach
# Delete or comment out the existing detect_and_handle_outliers function and its calls

# MAIN PROCESSING SECTION
# ----------------------

# Make sure the features list is defined before using it
# This should match the features list from your original code
features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
            'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature',
            'Solar Mean', 'Solar Max', 'Solar Min', 'Phytoplankton', 'Occurrences']

# Check which features are actually available in the dataframe
available_features = [f for f in features if f in df.columns]
if len(available_features) < len(features):
    print(f"Warning: Some features not found. Using {len(available_features)} available features:")
    print(available_features)
    features = available_features

# Minimal preprocessing (best for GBR)
print("Applying minimal preprocessing suitable for Gradient Boosting Regression...")
# Apply to features only, not target
minimal_preprocessing(df, features)

# Features and target
features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
            'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature',
            'Solar Mean', 'Solar Max', 'Solar Min', 'Phytoplankton', 'Occurrences']

# Check if all features exist in the dataframe
available_features = [f for f in features if f in df.columns]
if len(available_features) < len(features):
    print(f"Warning: Some features not found. Using {len(available_features)} available features:")
    print(available_features)
    features = available_features

target = 'Chlorophyll-a (ug/L)'

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(df[target], kde=True)
plt.title(f'Distribution of {target}')
plt.savefig('target_distribution.png')
plt.close()

# Log transform target if skewed
skewness = stats.skew(df[target])
print(f"Skewness of {target}: {skewness}")

if abs(skewness) > 1:
    print(f"Target is skewed (skewness={skewness}). Applying log transformation.")
    # Add small constant to handle zeros
    df[f'log_{target}'] = np.log1p(df[target])
    use_log = True
    log_target = f'log_{target}'
else:
    use_log = False
    log_target = target

# ENHANCEMENT 4: Advanced Feature Engineering
print("\n----- ENHANCED FEATURE ENGINEERING -----")

# Create lagged features
for feature in features + [target]:
    for lag in [1, 2, 3]:
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

# Create rolling mean and standard deviation features
for feature in features + [target]:
    for window in [3, 7, 14]:
        df[f'{feature}_roll_mean{window}'] = df[feature].rolling(window=window).mean()
        df[f'{feature}_roll_std{window}'] = df[feature].rolling(window=window).std()

# Create interaction terms between all pairs of features
print("Creating interaction features...")
for i, feat1 in enumerate(features):
    for feat2 in features[i + 1:]:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f'{feat1.split(" ")[0]}_{feat2.split(" ")[0]}_interact'
            df[interaction_name] = df[feat1] * df[feat2]
            print(f"  - Created {interaction_name}")

# Create polynomial features for key parameters
print("Creating polynomial features...")
for feature in features:
    if feature in df.columns:
        df[f'{feature}_squared'] = df[feature] ** 2
        print(f"  - Created {feature}_squared")

# Month as a proxy for seasonality if Date column exists
if 'Date' in df.columns:
    print("Creating seasonal features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Create seasonal features using sine and cosine transforms
    df['Season_sin'] = np.sin(df['DayOfYear'] * (2 * np.pi / 365))
    df['Season_cos'] = np.cos(df['DayOfYear'] * (2 * np.pi / 365))

    # One-hot encode month for better representation of seasonality
    month_dummies = pd.get_dummies(df['Month'], prefix='month')
    df = pd.concat([df, month_dummies], axis=1)

# Drop rows with NaN due to lag/rolling features
original_rows = len(df)
df = df.dropna()
print(f"Dropped {original_rows - len(df)} rows due to NaN values from feature engineering")

# All features including engineered ones
# Exclude any non-numeric columns, Date, target and log_target
exclude_cols = [col for col in df.columns if df[col].dtype == 'object'] + ['Date', target, log_target]
all_features = [col for col in df.columns if col not in exclude_cols]

# Verify all features are numeric
for col in all_features:
    if not np.issubdtype(df[col].dtype, np.number):
        print(f"Warning: Non-numeric column detected: {col}, type: {df[col].dtype}")
        all_features.remove(col)

# Print feature list
print(f"Total features after engineering: {len(all_features)}")

# Prepare data
X = df[all_features]
y = df[target]
y_log = df[log_target] if use_log else y

# Double-check that X contains only numeric data
print("X data types:")
print(X.dtypes.value_counts())

# ENHANCEMENT 5: Proper Time Series Cross-Validation
print("\n----- ENHANCED TIME SERIES CROSS-VALIDATION -----")
# Split data with proper time series validation
print("Setting up time series cross-validation...")
tscv = TimeSeriesSplit(n_splits=5)

# Splitting data into train and test
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
y_log_train, y_log_test = y_log.iloc[:train_size], y_log.iloc[train_size:]

print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Create a visualization to show time series CV splits
plt.figure(figsize=(10, 6))
for i, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    plt.scatter(train_idx, [i] * len(train_idx), c='blue', s=1, label='Train' if i == 0 else "")
    plt.scatter(test_idx, [i] * len(test_idx), c='red', s=1, label='Validation' if i == 0 else "")
plt.legend()
plt.title('Time Series Cross-Validation Splits')
plt.xlabel('Sample Index')
plt.ylabel('CV Iteration')
plt.tight_layout()
plt.savefig('time_series_cv_splits.png')
plt.close()


# Helper function to evaluate and plot results
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} - MAE: {mae:.4f}")
    print(f"{model_name} - RMSE: {rmse:.4f}")
    print(f"{model_name} - R²: {r2:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title(f"Chlorophyll-a Prediction - {model_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}")
    plt.xlabel("Samples")
    plt.ylabel("Chlorophyll-a (ug/L)")
    plt.legend()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}.png')
    plt.close()

    return y_pred, r2, rmse


# Replace the existing train-test split code with this date-based approach
print("\n----- ENHANCED DATE-BASED TRAIN-TEST SPLIT -----")

# Ensure Date column exists and is in datetime format
if 'Date' in df.columns:
    # Convert to datetime if not already
    df['Date'] = pd.to_datetime(df['Date'])

    # Get unique dates and sort
    unique_dates = df['Date'].sort_values().unique()

    # Calculate the split point - use the 80% mark of unique dates
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    print(f"Train-Test split date: {split_date}")
    print(f"Training data from: {unique_dates[0]} to {unique_dates[split_idx - 1]}")
    print(f"Testing data from: {unique_dates[split_idx]} to {unique_dates[-1]}")

    # Create train and test sets based on dates
    train_mask = df['Date'] < split_date
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]
    y_log_train = y_log[train_mask] if use_log else y_train
    y_log_test = y_log[~train_mask] if use_log else y_test

    print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

    # Visualize the train-test split based on dates
    plt.figure(figsize=(12, 6))
    plt.scatter(df.loc[train_mask, 'Date'], df.loc[train_mask, target],
                c='blue', alpha=0.6, label='Training data')
    plt.scatter(df.loc[~train_mask, 'Date'], df.loc[~train_mask, target],
                c='red', alpha=0.6, label='Testing data')
    plt.axvline(x=split_date, color='k', linestyle='--', label=f'Split date: {split_date.date()}')
    plt.title('Date-based Train-Test Split Visualization', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(target, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('date_based_split.png')
    plt.close()
else:
    print("Warning: 'Date' column not found. Using index-based split instead.")
    # Fall back to the original index-based split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    y_log_train, y_log_test = y_log.iloc[:train_size], y_log.iloc[train_size:]
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ENHANCEMENT 6: Feature Selection using Feature Importance
print("\n----- ENHANCED FEATURE SELECTION -----")
# Use a more efficient approach than RFECV - Random Forest feature importance
print("Performing feature selection using Random Forest feature importance...")
feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
feature_selector.fit(X_train, y_train if not use_log else y_log_train)

# Get feature importances
importances = feature_selector.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Print top features
print("Top 15 important features:")
print(feature_importance.head(15))

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Select top features - keep top 30% or at least 20 features
num_features = max(int(len(X_train.columns) * 0.3), 20)
num_features = min(num_features, len(X_train.columns))  # Ensure we don't select more than we have
selected_features = feature_importance['feature'][:num_features].values
print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features")

# Apply a threshold on feature importance as an alternative
# threshold = 0.01  # This means we keep features that contribute at least 1% to the model
# selected_features = feature_importance[feature_importance['importance'] > threshold]['feature'].values
# print(f"Selected {len(selected_features)} features with importance > {threshold}")

# Use selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Use selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# ENHANCEMENT 7: Hyperparameter Tuning
print("\n----- HYPERPARAMETER TUNING -----")

# 1. Random Forest Hyperparameter Tuning
print("\nTuning Random Forest hyperparameters...")
# Parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Use RandomizedSearchCV for efficiency
rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Fit the random search
rf_random.fit(X_train_selected, y_train if not use_log else y_log_train)

# Best parameters
print("Best Random Forest parameters found:")
for param, value in rf_random.best_params_.items():
    print(f"  - {param}: {value}")

# Train RF with best parameters
best_rf = rf_random.best_estimator_
print("\nTraining Random Forest with best parameters...")

# 2. Gradient Boosting Hyperparameter Tuning
print("\nTuning Gradient Boosting hyperparameters...")
# Parameter grid for Gradient Boosting
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Use RandomizedSearchCV for efficiency
gb_random = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=gb_param_grid,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Fit the random search
gb_random.fit(X_train_selected, y_train if not use_log else y_log_train)

# Best parameters
print("Best Gradient Boosting parameters found:")
for param, value in gb_random.best_params_.items():
    print(f"  - {param}: {value}")

# Train GB with best parameters
best_gb = gb_random.best_estimator_
print("\nTraining Gradient Boosting with best parameters...")

# 3. Create Pipeline with Robust Scaling for Gradient Boosting
gb_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', best_gb)
])

# Fit the pipeline
gb_pipeline.fit(X_train_selected, y_train if not use_log else y_log_train)

# Predictions
print("\nMaking predictions with best models...")
# Random Forest
rf_pred = best_rf.predict(X_test_selected)
if use_log:
    rf_pred = np.expm1(rf_pred)  # Convert back from log

# Gradient Boosting
gb_pred = gb_pipeline.predict(X_test_selected)
if use_log:
    gb_pred = np.expm1(gb_pred)  # Convert back from log

# Evaluate models
print("\nEvaluating models on test set...")
rf_pred, rf_r2, rf_rmse = evaluate_model(y_test, rf_pred, "Random Forest (Tuned)")
gb_pred, gb_r2, gb_rmse = evaluate_model(y_test, gb_pred, "Gradient Boosting (Tuned)")

# ENHANCEMENT 8: Feature Importance Analysis
print("\n----- FEATURE IMPORTANCE ANALYSIS -----")
# Get feature importances for Random Forest
rf_importances = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Random Forest feature importances:")
print(rf_importances.head(15))

# Get feature importances for Gradient Boosting
gb_importances = pd.DataFrame({
    'feature': selected_features,
    'importance': gb_pipeline.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Gradient Boosting feature importances:")
print(gb_importances.head(15))

# Plot feature importances
plt.figure(figsize=(14, 10))
plt.subplot(1, 2, 1)
sns.barplot(x='importance', y='feature', data=rf_importances.head(15))
plt.title('Random Forest Feature Importance')
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(x='importance', y='feature', data=gb_importances.head(15))
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()

plt.savefig('feature_importances_comparison.png')
plt.close()

# Compare model performance
plt.figure(figsize=(15, 8))
plt.plot(y_test, label='Actual', color='blue', linewidth=2)
plt.plot(rf_pred, label=f'Random Forest (R²={rf_r2:.3f})', linestyle='--')
plt.plot(gb_pred, label=f'Gradient Boosting (R²={gb_r2:.3f})', linestyle='--')
plt.title("Comparison of Tuned Models")
plt.xlabel("Samples")
plt.ylabel("Chlorophyll-a (ug/L)")
plt.legend()
plt.savefig('tuned_model_comparison.png')
plt.close()

# Print summary
print("\n===== SUMMARY OF RESULTS =====")
print(f"Tuned Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")
print(f"Tuned Gradient Boosting: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")

# ENHANCEMENT 9: Learning Curves
print("\n----- LEARNING CURVES -----")
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a plot of the learning curve for an estimator
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (neg MSE)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='neg_mean_squared_error')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# Plot learning curves
plot_learning_curve(
    best_rf, X_train_selected, y_train if not use_log else y_log_train,
    "Learning Curve - Random Forest", cv=tscv, n_jobs=-1
)
plt.savefig('rf_learning_curve.png')
plt.close()

plot_learning_curve(
    best_gb, X_train_selected, y_train if not use_log else y_log_train,
    "Learning Curve - Gradient Boosting", cv=tscv, n_jobs=-1
)
plt.savefig('gb_learning_curve.png')
plt.close()

# After making predictions, organize them by month
# Assuming df['Date'] exists in your test set
test_dates = df.iloc[train_size:]['Date']
test_results = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_test,
    'RF_Predicted': rf_pred,
    'GB_Predicted': gb_pred
})

# Convert Date to datetime if it isn't already
test_results['Date'] = pd.to_datetime(test_results['Date'])
test_results['Month'] = test_results['Date'].dt.month

# Print predictions by month
print("\n=== Monthly Predictions ===")
for month in range(1, 13):
    month_data = test_results[test_results['Month'] == month]
    if not month_data.empty:
        print(f"\nMonth {month}:")
        print("Actual values:", month_data['Actual'].values)
        print("Random Forest predictions:", month_data['RF_Predicted'].values)
        print("Gradient Boosting predictions:", month_data['GB_Predicted'].values)
        print("Number of predictions:", len(month_data))
        print("-" * 50)

# Final recommendations
print("\n===== FINAL RECOMMENDATIONS =====")
best_r2 = max(rf_r2, gb_r2)
if best_r2 <= 0.3:
    print("Model performance is still limited. Consider:")
    print("1. Collecting more data, especially in underrepresented conditions")
    print("2. Including additional environmental factors or external data sources")
    print("3. Exploring alternative models like neural networks or ensemble approaches")
    print("4. Consulting domain experts for specific chlorophyll-a drivers")
    print("5. Investigating temporal or spatial patterns that might require specialized models")
elif best_r2 <= 0.7:
    print("Model shows moderate predictive power. Consider:")
    print("1. Further feature engineering based on domain knowledge")
    print("2. Ensemble methods combining multiple models")
    print("3. Addressing any remaining data quality issues")
    print("4. Adding external data sources like weather or satellite observations")
else:
    print("Model shows strong predictive power. Consider:")
    print("1. Deploying model in production environment")
    print("2. Setting up monitoring system for model drift")
    print("3. Validating predictions against new real-world data")
    print("4. Developing an interactive tool for stakeholders")

# Identify best model
best_model_name = "Random Forest" if rf_r2 > gb_r2 else "Gradient Boosting"
print(f"\nBest model: {best_model_name} with R² = {best_r2:.4f}")

# Print top 5 most important features from best model
best_importances = rf_importances if rf_r2 > gb_r2 else gb_importances
print("\nTop 5 most important features:")
for i, row in best_importances.head(5).iterrows():
    print(f"- {row['feature']}: {row['importance']:.4f}")

# ENHANCEMENT 10: Enhanced Visualization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error


def visualize_model_results(y_true, y_pred_rf, y_pred_gb, rf_importances, gb_importances, top_n=10):
    """
    Create and display comprehensive visualizations for model results
    """
    # Set up the style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

    # Create a figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('Enhanced Model Analysis', fontsize=22, y=0.98)

    # 1. Time Series - Actual vs Predicted
    ax1 = axes[0, 0]
    sample_indices = range(len(y_true))

    ax1.plot(sample_indices, y_true, 'b-', label='Actual', linewidth=2)
    ax1.plot(sample_indices, y_pred_rf, 'r--', label=f'RF (R²={r2_score(y_true, y_pred_rf):.3f})', linewidth=2)
    ax1.plot(sample_indices, y_pred_gb, 'g--', label=f'GB (R²={r2_score(y_true, y_pred_gb):.3f})', linewidth=2)

    ax1.set_title(
        f'Actual vs Predicted Chlorophyll-a Levels',
        fontsize=16)
    ax1.set_xlabel('Sample Index', fontsize=14)
    ax1.set_ylabel('Chlorophyll-a (ug/L)', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Scatter Plot - Actual vs Predicted - Random Forest
    ax2 = axes[0, 1]

    ax2.scatter(y_true, y_pred_rf, alpha=0.6, color='darkred')

    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred_rf))
    max_val = max(max(y_true), max(y_pred_rf))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)

    ax2.set_title(
        f'Random Forest: Actual vs Predicted\nR² = {r2_score(y_true, y_pred_rf):.3f}, RMSE = {np.sqrt(mean_squared_error(y_true, y_pred_rf)):.3f}',
        fontsize=16)
    ax2.set_xlabel('Actual Chlorophyll-a (ug/L)', fontsize=14)
    ax2.set_ylabel('Predicted Chlorophyll-a (ug/L)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. Scatter Plot - Actual vs Predicted - Gradient Boosting
    ax3 = axes[0, 2]

    ax3.scatter(y_true, y_pred_gb, alpha=0.6, color='darkgreen')

    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred_gb))
    max_val = max(max(y_true), max(y_pred_gb))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=2)

    ax3.set_title(
        f'Gradient Boosting: Actual vs Predicted\nR² = {r2_score(y_true, y_pred_gb):.3f}, RMSE = {np.sqrt(mean_squared_error(y_true, y_pred_gb)):.3f}',
        fontsize=16)
    ax3.set_xlabel('Actual Chlorophyll-a (ug/L)', fontsize=14)
    ax3.set_ylabel('Predicted Chlorophyll-a (ug/L)', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 4. RF Feature Importance - Horizontal Bar Chart
    ax4 = axes[1, 0]

    # Use only top N features
    top_rf_features = rf_importances.head(top_n)

    # Create horizontal bar chart
    bars = ax4.barh(top_rf_features['feature'][::-1], top_rf_features['importance'][::-1], color='darkred')

    ax4.set_title('Top Random Forest Feature Importances', fontsize=16)
    ax4.set_xlabel('Importance Score', fontsize=14)
    ax4.set_ylabel('Feature', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax4.text(width * 1.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                 ha='left', va='center', fontsize=10)

    # 5. GB Feature Importance - Horizontal Bar Chart
    ax5 = axes[1, 1]

    # Use only top N features
    top_gb_features = gb_importances.head(top_n)

    # Create horizontal bar chart
    bars = ax5.barh(top_gb_features['feature'][::-1], top_gb_features['importance'][::-1], color='darkgreen')

    ax5.set_title('Top Gradient Boosting Feature Importances', fontsize=16)
    ax5.set_xlabel('Importance Score', fontsize=14)
    ax5.set_ylabel('Feature', fontsize=14)
    ax5.grid(True, alpha=0.3, axis='x')

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax5.text(width * 1.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}',
                 ha='left', va='center', fontsize=10)

    # 6. Residuals Comparison
    ax6 = axes[1, 2]

    # Calculate residuals
    residuals_rf = y_true - y_pred_rf
    residuals_gb = y_true - y_pred_gb

    # Create KDE plots
    sns.kdeplot(residuals_rf, ax=ax6, color='darkred', label='Random Forest', fill=True, alpha=0.3)
    sns.kdeplot(residuals_gb, ax=ax6, color='darkgreen', label='Gradient Boosting', fill=True, alpha=0.3)

    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.7, linewidth=2)
    ax6.set_title('Distribution of Residuals', fontsize=16)
    ax6.set_xlabel('Residual Value (Actual - Predicted)', fontsize=14)
    ax6.set_ylabel('Density', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add descriptive statistics for residuals
    rf_stats = f"RF - Mean: {np.mean(residuals_rf):.3f}, Std: {np.std(residuals_rf):.3f}"
    gb_stats = f"GB - Mean: {np.mean(residuals_gb):.3f}, Std: {np.std(residuals_gb):.3f}"

    ax6.text(0.95, 0.95, rf_stats + "\n" + gb_stats,
             transform=ax6.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig('enhanced_model_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


# Run the comprehensive visualization at the end
visualize_model_results(
    y_true=y_test,
    y_pred_rf=rf_pred,
    y_pred_gb=gb_pred,
    rf_importances=rf_importances,
    gb_importances=gb_importances,
    top_n=10
)

print("\n----- SAVING MODEL TO DISK -----")
import pickle

# Save the gradient boosting pipeline model
with open('chlorophyll_gb_model.pkl', 'wb') as file:
    pickle.dump(gb_pipeline, file)
print("Gradient Boosting pipeline saved as 'chlorophyll_gb_model.pkl'")

# Save selected features for future reference
with open('selected_features.pkl', 'wb') as file:
    pickle.dump(selected_features, file)
print(f"Selected features list saved as 'selected_features.pkl'")

# Optional: Save model metadata
model_metadata = {
    'r2_score': gb_r2,
    'rmse': gb_rmse,
    'use_log_transform': use_log,
    'feature_importances': gb_importances.to_dict(),
    'model_parameters': gb_pipeline.named_steps['model'].get_params(),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
}

with open('chlorophyll_gb_model_metadata.pkl', 'wb') as file:
    pickle.dump(model_metadata, file)
print("Model metadata saved as 'chlorophyll_gb_model_metadata.pkl'")

# Example of how to load the model later
print("\nTo load this model in the future, use:")
print("with open('chlorophyll_gb_model.pkl', 'rb') as file:")
print("    loaded_model = pickle.load(file)")
print("predictions = loaded_model.predict(X_new)")

# Replace the station-based forecasting section in your code with this fully fixed version

print("\n----- STATION-BASED FORECASTING USING ORIGINAL EXCEL FILE -----")

try:
    # Load the original Excel file
    original_df = pd.read_excel('merged_stations.xlsx')

    # Check if Station column exists
    if 'Station' not in original_df.columns:
        print("Error: 'Station' column not found in the Excel file.")
    else:
        # Convert Date to datetime
        if 'Date' in original_df.columns:
            original_df['Date'] = pd.to_datetime(original_df['Date'])
        else:
            print("Warning: No 'Date' column found in the Excel file. Using index as date.")
            original_df['Date'] = pd.date_range(start='2020-01-01', periods=len(original_df), freq='D')

        # Convert all numeric columns to float, handling errors
        print("Converting columns to appropriate data types...")
        for col in features + [target]:
            if col in original_df.columns:
                try:
                    original_df[col] = pd.to_numeric(original_df[col], errors='coerce')
                    print(f"  - Converted '{col}' to numeric type")
                except Exception as e:
                    print(f"  - Warning: Could not convert '{col}' to numeric: {e}")

        # Make sure target exists
        if target not in original_df.columns:
            print(f"Error: Target variable '{target}' not found in the Excel file.")
        else:
            # Find the last date in our dataset
            last_date = original_df['Date'].max()
            print(f"Last date in dataset: {last_date.date()}")

            # Get unique stations and print them
            stations = original_df['Station'].unique()
            num_stations = len(stations)
            print(f"Found {num_stations} unique stations:")
            for i, station in enumerate(stations, 1):
                print(f"  {i}. {station}")

            # DEBUG: Print the count of records per station to check for inconsistencies
            station_counts = original_df['Station'].value_counts()
            print("\nStation record counts:")
            print(station_counts)

            # Check for stations that only appear in certain months
            print("\nChecking station appearance by month...")
            month_station_matrix = pd.crosstab(original_df['Date'].dt.month, original_df['Station'])
            print(month_station_matrix)

            # Option to limit to a fixed set of stations if needed
            use_fixed_stations = True  # Set to True to use only a specific set of stations

            if use_fixed_stations:
                # Identify the most common stations (with data in most months)
                station_month_counts = month_station_matrix.sum(axis=0)
                sorted_stations = station_month_counts.sort_values(ascending=False)
                print("\nStations by month coverage:")
                print(sorted_stations)

                # Select the top 9 stations (or another fixed number)
                top_stations = sorted_stations.index[:9].tolist()
                print(f"\nUsing top 9 stations for consistent forecasting: {top_stations}")
                stations = top_stations

            # Generate dates for the next 18 months
            raw_forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                               periods=18, freq='M')

            print(f"Raw forecast period: {raw_forecast_dates.min().date()} to {raw_forecast_dates.max().date()}")

            # ADDITIONAL FIX: Limit to having each month appear exactly once in the forecast
            # This prevents the issue of having some months appear twice
            unique_months = []
            unique_dates = []

            for date in raw_forecast_dates:
                month = date.month
                if month not in unique_months:
                    unique_months.append(month)
                    unique_dates.append(date)

            # Use only the unique months for forecasting
            forecast_dates = pd.DatetimeIndex(unique_dates)
            print(f"Using {len(forecast_dates)} unique months for forecasting")
            print(f"Final forecast period covers these months: {sorted(unique_months)}")

            # For full calendar year, ensure we have all 12 months
            if len(unique_months) < 12:
                missing_months = set(range(1, 13)) - set(unique_months)
                print(f"Warning: Missing months in forecast period: {missing_months}")
                print("Will use available months only.")

            # Train models on the preprocessed data (keeping the same model as before)
            # This ensures we use all the feature engineering that was done
            print("Training models on the processed data...")

            # Use the best model configurations from our previous hyperparameter tuning
            final_rf = RandomForestRegressor(**rf_random.best_params_, random_state=42)
            final_rf.fit(X[selected_features], y if not use_log else y_log)

            final_gb_pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', GradientBoostingRegressor(**gb_random.best_params_, random_state=42))
            ])
            final_gb_pipeline.fit(X[selected_features], y if not use_log else y_log)

            print("Models successfully trained.")

            # Calculate monthly averages for each feature by station
            print("Calculating station-specific monthly averages...")
            station_monthly_avgs = {}

            for station in stations:
                station_df = original_df[original_df['Station'] == station]

                # Skip if station has no data
                if len(station_df) == 0:
                    print(f"Warning: No data for station {station}, skipping...")
                    continue

                station_monthly_avgs[station] = {}

                # Calculate monthly averages for features and target for this station
                for feature in features + [target]:
                    if feature in station_df.columns:
                        try:
                            # Calculate monthly averages for this station, dropping NaN values first
                            monthly_avg = station_df.groupby(station_df['Date'].dt.month)[feature].apply(
                                lambda x: x.dropna().mean())
                            station_monthly_avgs[station][feature] = monthly_avg

                            # If there are months with no data, fill them using overall average for that month
                            if len(monthly_avg) < 12:
                                print(f"  - Warning: Station {station} missing data for some months for {feature}")
                                # Get overall monthly average across all stations
                                overall_monthly_avg = original_df.groupby(original_df['Date'].dt.month)[feature].apply(
                                    lambda x: x.dropna().mean())

                                # Fill missing months with overall average
                                for month in range(1, 13):
                                    if month not in monthly_avg.index:
                                        if month in overall_monthly_avg.index:
                                            station_monthly_avgs[station][feature][month] = overall_monthly_avg[month]
                                        else:
                                            # If still no data, use overall average
                                            station_monthly_avgs[station][feature][month] = station_df[
                                                feature].dropna().mean()
                        except Exception as e:
                            print(f"  - Error calculating monthly averages for {station}, {feature}: {e}")
                            print("    Trying alternative approach...")
                            # Use a more robust approach
                            try:
                                # Create a month column
                                station_df_month = station_df.copy()
                                station_df_month['Month'] = station_df['Date'].dt.month

                                # Group manually and calculate mean
                                monthly_values = {}
                                for month in range(1, 13):
                                    month_data = station_df_month[station_df_month['Month'] == month][feature].dropna()
                                    if len(month_data) > 0:
                                        monthly_values[month] = month_data.astype(float).mean()

                                # Convert to Series
                                if monthly_values:
                                    monthly_avg = pd.Series(monthly_values)
                                    station_monthly_avgs[station][feature] = monthly_avg
                                else:
                                    # Fallback to overall mean
                                    overall_mean = original_df[feature].dropna().astype(float).mean()
                                    station_monthly_avgs[station][feature] = pd.Series(
                                        {month: overall_mean for month in range(1, 13)})
                            except Exception as e2:
                                print(f"    Second approach also failed: {e2}")
                                # Use a constant value as last resort
                                try:
                                    overall_mean = original_df[feature].dropna().astype(float).mean()
                                    if pd.isna(overall_mean):
                                        overall_mean = 0
                                except:
                                    overall_mean = 0
                                station_monthly_avgs[station][feature] = pd.Series(
                                    {month: overall_mean for month in range(1, 13)})


            # Create a function to convert station name to one-hot encoded features
            def create_station_dummies(station_name, all_stations):
                """Create one-hot encoded features for a station"""
                dummies = {}
                for s in all_stations:
                    col_name = f"station_{s.replace(' ', '_')}"  # This matches how the dummies were likely created in preprocessing
                    dummies[col_name] = 1 if s == station_name else 0
                return dummies


            # Prepare to store all forecasts
            all_forecasts = []

            # For each station - guaranteed to be consistent now with our fixed station list
            for station in stations:
                print(f"\nGenerating forecasts for station: {station}")

                # Skip if we don't have monthly averages for this station
                if station not in station_monthly_avgs:
                    print(f"Skipping station {station} - no monthly data available")
                    continue

                # Create a future DataFrame for this station with all dates
                # IMPORTANT: Create exactly one row per forecast date
                future_station_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Station': [station] * len(forecast_dates)  # Ensure station is consistent
                })

                # Number of predictions for this station
                num_predictions = len(future_station_df)
                print(f"Creating {num_predictions} predictions for station {station}")

                # Extract month and seasonal features from the future dates
                future_station_df['Month'] = future_station_df['Date'].dt.month
                future_station_df['Year'] = future_station_df['Date'].dt.year
                future_station_df['DayOfYear'] = future_station_df['Date'].dt.dayofyear
                future_station_df['Season_sin'] = np.sin(future_station_df['DayOfYear'] * (2 * np.pi / 365))
                future_station_df['Season_cos'] = np.cos(future_station_df['DayOfYear'] * (2 * np.pi / 365))

                # Create month dummy variables
                month_dummies = pd.get_dummies(future_station_df['Month'], prefix='month')
                # Convert boolean to integer
                month_dummies = month_dummies.astype(int)
                future_station_df = pd.concat([future_station_df, month_dummies], axis=1)

                # Instead of using station dummies, just keep the station name as an identifier
                # We're not using it as a feature in the model
                print(f"Processing station {station} without creating station dummy variables")

                # Create station dummy variables - need to make sure column names match exactly what's in X
                # Get the actual station dummy columns from X
                station_cols_in_X = [col for col in X.columns if col.startswith('station_')]

                # Apply monthly averages to the future dataframe for this station
                for feature in features:
                    if feature in station_monthly_avgs[station]:
                        future_station_df[feature] = future_station_df['Month'].map(
                            station_monthly_avgs[station][feature])

                # Get most recent values from historical data for this station
                station_recent_df = original_df[original_df['Station'] == station].sort_values('Date')

                # If this station has data, get recent values
                recent_values = {}
                if len(station_recent_df) > 0:
                    for feature in features + [target]:
                        if feature in station_recent_df.columns:
                            # Convert to numeric and drop NaN values
                            feature_values = pd.to_numeric(station_recent_df[feature], errors='coerce').dropna()
                            if len(feature_values) >= 3:
                                recent_values[feature] = feature_values.iloc[-3:].values  # Get last 3 values
                            elif len(feature_values) > 0:
                                # Use whatever data we have (less than 3 points)
                                recent_values[feature] = feature_values.iloc[-len(feature_values):].values
                            else:
                                # Use zeros as fallback
                                recent_values[feature] = np.zeros(3)
                        else:
                            # Use zeros as fallback
                            recent_values[feature] = np.zeros(3)
                else:
                    # If no data for this station, initialize with zeros
                    for feature in features + [target]:
                        recent_values[feature] = np.zeros(3)

                # Initialize array to store predictions for this station
                rf_future_preds = np.zeros(len(future_station_df))
                gb_future_preds = np.zeros(len(future_station_df))

                # Iterative forecasting - one month at a time
                print(f"Generating month-by-month forecasts for station {station}...")

                for i in range(len(future_station_df)):
                    # Create lag features based on recent predictions or historical values
                    for feature in features + [target]:
                        if feature in original_df.columns and feature in station_monthly_avgs[station]:
                            # For the first few predictions, use historical data
                            for lag in [1, 2, 3]:
                                lag_col = f'{feature}_lag{lag}'
                                if lag <= i:
                                    # Use previously predicted values for future months
                                    if feature == target:
                                        future_station_df.loc[future_station_df.index[i], lag_col] = gb_future_preds[
                                            i - lag]
                                    else:
                                        # Use the corresponding month from previous year as an approximation
                                        month_val = station_monthly_avgs[station][feature][
                                            future_station_df.loc[future_station_df.index[i], 'Month']]
                                        future_station_df.loc[future_station_df.index[i], lag_col] = month_val
                                else:
                                    # Use historical values for initial lags
                                    idx = -lag + i
                                    if idx >= -len(recent_values[feature]):
                                        future_station_df.loc[future_station_df.index[i], lag_col] = \
                                            recent_values[feature][idx]
                                    else:
                                        # Not enough history, use monthly average
                                        month_val = station_monthly_avgs[station][feature][
                                            future_station_df.loc[future_station_df.index[i], 'Month']]
                                        future_station_df.loc[future_station_df.index[i], lag_col] = month_val

                    # Create rolling means and stds
                    # For simplicity, we'll approximate these with the monthly averages initially
                    for feature in features + [target]:
                        if feature in original_df.columns and feature in station_monthly_avgs[station]:
                            for window in [3, 7, 14]:
                                mean_col = f'{feature}_roll_mean{window}'
                                std_col = f'{feature}_roll_std{window}'

                                # For initial predictions, use average of historical values for that month
                                month_val = station_monthly_avgs[station][feature][
                                    future_station_df.loc[future_station_df.index[i], 'Month']]
                                future_station_df.loc[future_station_df.index[i], mean_col] = month_val

                                # For std, use the std of that feature in the historical dataset for the station & month
                                month_data = pd.to_numeric(station_recent_df[station_recent_df['Date'].dt.month ==
                                                                             future_station_df.loc[
                                                                                 future_station_df.index[i], 'Month']][
                                                               feature],
                                                           errors='coerce').dropna()

                                if len(month_data) > 1:
                                    month_std = month_data.std()
                                    future_station_df.loc[
                                        future_station_df.index[i], std_col] = month_std if not np.isnan(
                                        month_std) else 0
                                else:
                                    # Not enough data for std calculation, use overall std
                                    all_data = pd.to_numeric(station_recent_df[feature], errors='coerce').dropna()
                                    future_station_df.loc[future_station_df.index[i], std_col] = all_data.std() if len(
                                        all_data) > 1 else 0

                    # Create interaction features
                    for j, feat1 in enumerate(features):
                        if feat1 in future_station_df.columns:
                            for feat2 in features[j + 1:]:
                                if feat2 in future_station_df.columns:
                                    interaction_name = f'{feat1.split(" ")[0]}_{feat2.split(" ")[0]}_interact'
                                    future_station_df.loc[future_station_df.index[i], interaction_name] = \
                                        future_station_df.loc[future_station_df.index[i], feat1] * \
                                        future_station_df.loc[
                                            future_station_df.index[i], feat2]

                    # Create polynomial features
                    for feature in features:
                        if feature in future_station_df.columns:
                            future_station_df.loc[future_station_df.index[i], f'{feature}_squared'] = \
                                future_station_df.loc[future_station_df.index[i], feature] ** 2

                    # Create a feature vector for this future date
                    row_idx = future_station_df.index[i]
                    # First, create dict with all selected features initialized to NaN
                    current_features = {col: np.nan for col in selected_features}

                    # Then fill in the values we have
                    for col in selected_features:
                        if col in future_station_df.columns:
                            current_features[col] = future_station_df.loc[row_idx, col]

                    # Convert to DataFrame with the right structure
                    X_future_i = pd.DataFrame([current_features])

                    # Fill any missing values with median from training data
                    for col in X_future_i.columns:
                        if X_future_i[col].isna().any():
                            if col in X.columns:
                                median_val = X[col].median()
                                X_future_i[col] = X_future_i[col].fillna(median_val)
                            else:
                                X_future_i[col] = 0  # Default to 0 if we don't have historical data

                    # Make predictions for this month
                    try:
                        # Check that all required features are available
                        missing_features = [feat for feat in selected_features if feat not in X_future_i.columns]
                        if missing_features:
                            raise ValueError(f"Missing required features: {missing_features}")

                        rf_pred = final_rf.predict(X_future_i[selected_features])[0]
                        gb_pred = final_gb_pipeline.predict(X_future_i[selected_features])[0]

                        # Convert back from log if needed
                        if use_log:
                            rf_pred = np.expm1(rf_pred)
                            gb_pred = np.expm1(gb_pred)

                        # Cap negative predictions at 0 (can't have negative chlorophyll)
                        rf_pred = max(0, rf_pred)
                        gb_pred = max(0, gb_pred)

                        # Store predictions
                        rf_future_preds[i] = rf_pred
                        gb_future_preds[i] = gb_pred

                    except Exception as e:
                        print(f"Error making prediction for station {station}, month {i + 1}: {e}")
                        # Use last valid prediction or 0
                        rf_pred = rf_future_preds[i - 1] if i > 0 else 0
                        gb_pred = gb_future_preds[i - 1] if i > 0 else 0
                        rf_future_preds[i] = rf_pred
                        gb_future_preds[i] = gb_pred

                # Double-check prediction count
                if len(rf_future_preds) != len(forecast_dates):
                    print(
                        f"WARNING: Expected {len(forecast_dates)} predictions for station {station}, but got {len(rf_future_preds)}")

                # Create a DataFrame with the results for this station
                forecast_results = pd.DataFrame({
                    'Date': future_station_df['Date'],
                    'Year': future_station_df['Year'],
                    'Month': future_station_df['Month'],
                    'Station': station,
                    'Month_Name': future_station_df['Date'].dt.month_name(),
                    'RF_Predicted': rf_future_preds,
                    'GB_Predicted': gb_future_preds,
                    'Ensemble_Predicted': (rf_future_preds + gb_future_preds) / 2  # Average of both models
                })

                print(f"Completed forecasts for station {station} - generated {len(forecast_results)} predictions")
                all_forecasts.append(forecast_results)

            # Combine all station forecasts
            combined_forecast = pd.concat(all_forecasts, ignore_index=True)

            # Check for consistency in prediction counts per month
            print("\nVerifying forecast consistency:")
            forecast_counts = combined_forecast.groupby('Month')['Station'].count().reset_index()
            forecast_counts.columns = ['Month', 'Number of predictions']
            print(forecast_counts)

            # If we have data across multiple years, also check by year-month
            if combined_forecast['Year'].nunique() > 1:
                print("\nVerifying forecast consistency by year-month:")
                year_month_counts = combined_forecast.groupby(['Year', 'Month'])['Station'].count().reset_index()
                year_month_counts.columns = ['Year', 'Month', 'Number of predictions']
                print(year_month_counts)

            # Check stations per month
            station_month_matrix = pd.crosstab(combined_forecast['Month'], combined_forecast['Station'])
            print("\nStation predictions per month:")
            print(station_month_matrix)

            # Check if we have consistent prediction counts
            expected_count = len(stations)
            is_consistent = forecast_counts['Number of predictions'].nunique() == 1 and \
                            forecast_counts['Number of predictions'].iloc[0] == expected_count

            if is_consistent:
                print(f"\nSUCCESS: Consistent prediction counts achieved ({expected_count} stations per month).")
            else:
                print("\nWARNING: Prediction counts are still inconsistent. Check the forecast data carefully.")

            print("\nAll Station Forecasts Summary:")
            summary = combined_forecast.groupby(['Month_Name', 'Station'])['Ensemble_Predicted'].mean().unstack()
            print(summary)

            # Save the forecast to CSV
            combined_forecast.to_csv('chlorophyll_station_forecast.csv', index=False)
            print("Forecast saved to 'chlorophyll_station_forecast.csv'")

            # Create a wide format version (one row per date, columns for each station)
            wide_forecast = combined_forecast.pivot(index='Date', columns='Station', values='Ensemble_Predicted')
            wide_forecast.reset_index(inplace=True)
            wide_forecast.to_csv('chlorophyll_station_forecast_wide.csv', index=False)
            print("Wide format forecast saved to 'chlorophyll_station_forecast_wide.csv'")

            # Plot the historical data and the forecast for each station
            for station in stations:
                plt.figure(figsize=(15, 8))

                # Get historical data for this station
                station_hist = original_df[original_df['Station'] == station]

                # Get forecast for this station
                station_forecast = combined_forecast[combined_forecast['Station'] == station]

                if len(station_hist) > 0:  # Only plot if we have historical data
                    # Plot historical data
                    plt.plot(station_hist['Date'], station_hist[target], 'b.-', label='Historical Data', alpha=0.7)

                    # Plot forecast
                    plt.plot(station_forecast['Date'], station_forecast['RF_Predicted'], 'r.--',
                             label='Random Forest Forecast', markersize=8, alpha=0.7)
                    plt.plot(station_forecast['Date'], station_forecast['GB_Predicted'], 'g.--',
                             label='Gradient Boosting Forecast', markersize=8, alpha=0.7)

                    # Add shaded area to indicate forecast region
                    if len(station_hist) > 0 and len(station_forecast) > 0:
                        plt.axvspan(station_hist['Date'].max(), station_forecast['Date'].max(),
                                    alpha=0.2, color='gray', label='Forecast Period')

                    plt.title(f'Station {station}: Chlorophyll-a Historical Data and Forecast', fontsize=16)
                    plt.xlabel('Date', fontsize=14)
                    plt.ylabel('Chlorophyll-a (ug/L)', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=12)

                    # Improve x-axis date formatting
                    plt.gcf().autofmt_xdate()

                    plt.tight_layout()
                    plt.savefig(f'chlorophyll_forecast_station_{station}.png', dpi=300)
                    plt.close()

            # Create a heatmap visualization of forecasts across all stations
            pivot_data = combined_forecast.pivot_table(
                index='Date',
                columns='Station',
                values='Ensemble_Predicted',
                aggfunc='mean'
            )

            plt.figure(figsize=(16, 10))
            sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=.5)
            plt.title('Forecasted Chlorophyll-a Levels Across All Stations', fontsize=16)
            plt.ylabel('Date', fontsize=14)
            plt.xlabel('Station', fontsize=14)
            plt.tight_layout()
            plt.savefig('chlorophyll_forecast_heatmap.png', dpi=300)
            plt.close()

            # Create a multi-line plot comparing all stations
            plt.figure(figsize=(16, 10))

            # For each station, plot a line
            for station in stations:
                station_data = combined_forecast[combined_forecast['Station'] == station]
                if len(station_data) > 0:
                    plt.plot(station_data['Date'], station_data['Ensemble_Predicted'],
                             'o-', label=f'Station {station}', linewidth=2, markersize=4)

            plt.title('Comparison of Forecasted Chlorophyll-a Across All Stations', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Chlorophyll-a (ug/L)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig('chlorophyll_forecast_comparison.png', dpi=300)
            plt.close()

            # Save the full model and forecasting pipeline for future use
            print("\nSaving the forecast models and results...")

            import pickle

            forecast_package = {
                'rf_model': final_rf,
                'gb_model': final_gb_pipeline,
                'selected_features': selected_features,
                'forecast_results': combined_forecast,
                'last_training_date': last_date,
                'forecast_dates': forecast_dates,
                'use_log_transform': use_log,
                'station_monthly_stats': station_monthly_avgs,
                'stations_used': stations  # Include the final list of stations used
            }

            with open('chlorophyll_station_forecast_package.pkl', 'wb') as file:
                pickle.dump(forecast_package, file)

            print("Forecast package saved as 'chlorophyll_station_forecast_package.pkl'")
            print("\nStation-based forecast process complete!")

except Exception as e:
    print(f"Error loading or processing Excel file: {e}")
    import traceback

    traceback.print_exc()
    print("\nDetailed error information above.")
