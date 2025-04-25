'''
Helper Functions for UK-DALE Appliance Energy Cost Prediction
'''

# =========================================== UK-DALE EDA =====================================


# To understand data structure
def print_hierarchy(d):
    """
    Recursively prints a clean, aligned hierarchy from a nested dictionary.
    Output visually reflects hierarchy levels with indentation and hash symbols.

    Returns:
        int: Total number of printed dictionary keys
    """
    def recurse(subtree, level, counter):
        cnt = 0
        for key in subtree:
            if isinstance(subtree[key], dict):
                hash_str = "#" * (level + 1)
                spacing = " " * (6 - len(hash_str))  # ensures alignment in the hash column
                indent = " " * (level * 2)           # actual indentation for hierarchy

                print(f"cnt_lvl={level:<2}  counter={counter:<2}   {hash_str}{spacing} {indent}- {key}")
                
                counter += 1
                sub_cnt, counter = recurse(subtree[key], level + 1, counter)
                cnt += sub_cnt + 1
        return cnt, counter

    total_cnt, _ = recurse(d, 0, 0)
    return total_cnt


# To convert power to energy
def calculate_energy_kwh(df, df_duration_hours):
    """
    Computes total energy in kWh (with sign preserved).

    Parameters:
        df (pd.DataFrame): DataFrame with index = datetime, values = power_readings in watts
        sample_seconds (float): Sampling interval in hours
    Returns:
        energy_kwh (pd.DataFrame): DataFrame with index = datetime, values = energy per month in kWh
    """
    energy_kwh = df.resample('M').sum() * df_duration_hours / 1000 # W to kW
    return  energy_kwh

# =========================================== Data Preprocessing =====================================
# Missing value: To add a column to help idenfy group gap duration
def add_gap_column(df, column_name='power_active', gap_time=6):
    """
    Adds a 'gap' column to the input DataFrame based on missing values
    in the specified column.

    Parameters:
        df (pd.DataFrame): Input DataFrame (e.g., 'test')
        column_name (str): Name of the column to check for nulls
        gap_time (int): Value to increment when null is found

    Returns:
        pd.DataFrame: A new DataFrame with an additional 'gap' column

    """
    import numpy as np
    import pandas as pd

    power_series = df[column_name].values
    gap = []
    group_id = []

    # Handle the first row
    if not np.isnan(power_series[0]):
        gap.append(0)
    else:
        gap.append(gap_time)

    group_id.append(0)

    # Loop through the rest
    for i in range(1, len(power_series)):
        if not np.isnan(power_series[i]):
            gap.append(0)
            if not np.isnan(power_series[i-1]):
                group_id.append(group_id[i - 1])
            else:
                group_id.append(group_id[i - 1] + 1)
        elif np.isnan(power_series[i]) and not np.isnan(power_series[i - 1]):
            gap.append(gap_time)
            group_id.append(group_id[i - 1] + 1)
        elif np.isnan(power_series[i]) and np.isnan(power_series[i - 1]):
            gap.append(gap[i - 1] + gap_time)
            group_id.append(group_id[i - 1])

    # Return a new dataframe with the gap column added
    df_result = df.copy()
    df_result['gap_acc'] = gap
    df_result['gap_group'] = group_id
    df_result['gap_max'] = df_result.groupby('gap_group')['gap_acc'].transform('max')
    
    return df_result

# Missing value: To treat missing values based on their group gap duration
def clean_power_data(df_rename, df_rename_ref, max_gap_minute=2, column_name='power_active'):
    """
    Cleans the 'power_active' column based on max_gap_minute and external reference DataFrame.
    
    Parameters:
        df_rename (pd.DataFrame): The DataFrame to clean.
        df_rename_ref (pd.DataFrame): The reference DataFrame with gap_max and power_active.
        max_gap_minute (int): Maximum allowed gap in minutes.
        column_name (str): The name of the column to clean, default is 'power_active'.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with the 'power_active' column updated.
    """
    df_clean = df_rename.copy()

    # Replace missing power_active with 0 if gap_max > max_gap_minute
    df_clean.loc[
        (df_rename_ref['gap_max'] > max_gap_minute * 60) & (df_rename_ref[column_name].isnull()),
        column_name
    ] = 0

    # Forward-fill and backfill the 'power_active' column
    df_clean[column_name] = df_clean[column_name].fillna(method='ffill').fillna(method='bfill')

    return df_clean



# =========================================== Model Evaluation =====================================
# For Multiple Plots
def plot_data_multiple_models(
    predictions_dict,  # dictionary: {'ModelName': y_pred}
    X_train, y_train, X_val, y_val,
    cycle, duration,  # e.g., 'Day', (start_time, end_time)
    iteration
):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot train and actual validation data
    # ax.plot(X_train.index, y_train, 'g-', label='Train')
    ax.plot(X_val.index, y_val, color='black', ls='-', label='Ground Truth')

    # Plot predictions from each model
    for model_name, y_pred in predictions_dict.items():
        ax.plot(X_val.index, y_pred, linestyle='--', label=f'Predicted {model_name}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Active Power of Fridge at Building 1 in Watts')
    ax.set_title(f'Actual VS Predicted Fridge Power Usage - {cycle} Pattern')

    ax.set_xlim(pd.to_datetime(duration[0]), pd.to_datetime(duration[1]))
    fig.autofmt_xdate()

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'./figures/final/B1_fridge_val_{iteration}_all_models_{cycle}.png', dpi=300)
    print(f"✅ Figure saved as ./figures/final/B1_fridge_val_{iteration}_all_models_{cycle}.png.")
    plt.show()

# For single plot
def plot_data(y_pred, X_fact, y_fact, model, cycle, duration, stage, iteration):
    
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(10,6))

    # ax.plot(X_train.index, y_train, 'g-', label='Train')
    ax.plot(X_fact.index, y_fact, color='black', ls='-', label='Ground Truth')
    ax.plot(X_fact.index, y_pred, 'r:', label=f'Predicted {model}')

    ax.set_xlabel('Date')
    ax.set_ylabel('Active Power of Fridge at Building 1 in Watts')
    plt.title(f'Actual VS Predicted Fridge Power Usage - {cycle} Pattern')

    fig.autofmt_xdate()

    # Zoom in to later part
    ax.set_xlim(pd.to_datetime(duration[0]), pd.to_datetime(duration[1]))

    # Plot the legend
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./figures/final/B1_fridge_{iteration}_{stage}_{model}_{cycle}.png', dpi=300)
    print(f"✅ Figure saved as ./figures/final/B1_fridge_{iteration}_{stage}_{model}_{cycle}.png.")



# To create rmse function
def root_mean_squared_error(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))

# To evaluate regressors using metrics
def evaluate_reg_models(regressors, X_val, y_val, metrics, train_times_str):
    """
    Evaluate named regressors using given metrics and convert training time.

    Parameters:
        regressors (dict): {model_name: trained regressor}
        X_val (DataFrame): validation features
        y_val (Series): validation targets
        metrics (list): list of metric functions
        train_times_str (list): list of string-formatted train times, one per model

    Returns:
        models (list): list of model names
        y_preds (dict): {model_name: list of predictions}
        metric_results (dict): {metric_name: list of scores per model}
        train_time (list): list of numeric training times in seconds
    """

    # Helper: convert time string like "10m7s" to seconds
    def parse_time_to_seconds(time_str):
        import re
        time_str = time_str.strip().lower()
        total_seconds = 0
        match = re.findall(r'(\d+\.?\d*)([hms])', time_str)
        for val, unit in match:
            val = float(val)
            if unit == 'h':
                total_seconds += val * 3600
            elif unit == 'm':
                total_seconds += val * 60
            elif unit == 's':
                total_seconds += val
        return total_seconds

    models = []
    y_preds = {}
    metric_results = {metric.__name__: [] for metric in metrics}

    for idx, (name, model) in enumerate(regressors.items()):
        y_pred = model.predict(X_val)
        y_preds[name] = y_pred.tolist()
        models.append(name)

        for metric in metrics:
            score = metric(y_val, y_pred)
            metric_results[metric.__name__].append(score)

    # Convert external time strings to seconds
    train_time = [parse_time_to_seconds(t) for t in train_times_str]

    return models, y_preds, metric_results, train_time


# =========================================== Result =====================================
# To convert power to energy cost
def calculate_energy_monthly_cost(hourly_power_watts, price_per_kwh):
    """
    Computes energy cost per month (with sign preserved).

    Parameters:
        hourly_power_watts (pd.DataFrame): DataFrame with index = datetime, values = power_readings in watts
        price_per_kwh (float): Price of electricity in kwh
    Returns:
        monthly_cost (pd.DataFrame): DataFrame with index = datetime, values = energy cost per month
    """
    hourly_energy_kwh = hourly_power_watts / 1000
    monthly_energy_kwh = hourly_energy_kwh.resample('M').sum()
    monthly_cost = monthly_energy_kwh * price_per_kwh
    return monthly_cost



# =========================================== Others =====================================

def print_start_end_date(df, year_col='year', month_col='month', day_col='day'):
    """
    Prints the start and end date of a DataFrame in YYYY-MM-DD format,
    based on the year, month, and day columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame
        year_col (str): Name of the column containing year
        month_col (str): Name of the column containing month
        day_col (str): Name of the column containing day
    """
    start_row = df.iloc[0]
    end_row = df.iloc[-1]

    start_date = f"{int(start_row[year_col])}-{int(start_row[month_col]):02d}-{int(start_row[day_col]):02d}"
    end_date = f"{int(end_row[year_col])}-{int(end_row[month_col]):02d}-{int(end_row[day_col]):02d}"

    print(f"start: {start_date}")
    print(f"end: {end_date}")



def build_datetime_col_ymd(df, year_col='year', month_col='month', day_col='day'):
    """
    Converts separate year, month, and day columns in a DataFrame into a pandas datetime Series.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        year_col (str): Column name for year
        month_col (str): Column name for month
        day_col (str): Column name for day

    Returns:
        pd.Series: A pandas Series of datetime objects
    """
    date_str = df[year_col].astype(str) + '-' + \
               df[month_col].astype(str).str.zfill(2) + '-' + \
               df[day_col].astype(str).str.zfill(2)

    return pd.to_datetime(date_str)