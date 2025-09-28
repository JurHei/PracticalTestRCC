import requests # for querying Baltic API
import pandas as pd # for handling tabular time-series data
import matplotlib.pyplot as plt # for creating graphs
import matplotlib.dates as mdates
import json
from datetime import datetime # handles data/time parsing
import numpy as np # deals with null values from API

# Configuration
BASE_URL = "https://api-baltic.transparency-dashboard.eu/api/v1/export"
PARAMS = {
    "start_date": "2025-09-21T00:00",  
    "end_date": "2025-09-22T00:00",
    "output_time_zone": "EET",
    "output_format": "json",
    "json_header_groups": "0"
}
HEADERS = {"accept": "application/json"}

#Fetching data and conversion to readable tables
def fetch_data(data_id):
    """Parse 'timeseries' structure."""
    country_names = ['Estonia', 'Latvia', 'Lithuania']
    params = PARAMS.copy() # creates copy of parameters to avoid modifiying initial parameters
    params["id"] = data_id # sets dataset to export
    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS) #sending full request to API
        response.raise_for_status() # checking the request status
        data = response.json() # conversion of data to JSON

        api_data = data.get('data', {}) # extracts actual content 
        columns = api_data.get('columns', []) #extracts columns array
        timeseries = api_data.get('timeseries', []) #extracts timeseries array        
        
        # Build DataFrame
        # Preparing a readable list of columns
        rows = []
        col_names = []
        for col in columns:
            # Create descriptive column name: country_direction_label
            country = col.get('group_level_0', 'Unknown')
            direction = col.get('group_level_1', '')
            label = col.get('label', '')
            col_name = f"{country}_{direction}_{label}".strip('_')
            col_names.append(col_name)
        
        #processing time series intervals into rows
        non_null_count = 0
        for ts_item in timeseries:
            from_ts = pd.to_datetime(ts_item['from']) #conversion to Pandas datetime object
            to_ts = pd.to_datetime(ts_item['to'])
            mid_ts = from_ts + (to_ts - from_ts) / 2  # Midpoint timestamp, (15 min resolution)
            values = ts_item.get('values', []) # extracts array of values

            # Building row
            row = {'timestamp': mid_ts}
            has_non_null = False    #check if row has valid data
            for i, val in enumerate(values):    #Assigning values to corresponding column name
                row[col_names[i]] = float(val) if val is not None else np.nan #conversion to float for numeric data
                if val is not None:
                    has_non_null = True
            #avoiding empty rows if row does not have data
            if has_non_null: 
                non_null_count += 1
                rows.append(row)
        
        #creating and preparing the dataframe
        df = pd.DataFrame(rows) #conversion of list of dicts into dataframe
        df.set_index('timestamp', inplace=True) #promotes timestamp column to index ???
        df = df.sort_index()  # Sort by time
        
         # Per-region processing
        if data_id == 'activations_afrr':
            # For each country, sum upward + downward activation columns (total MWh dispatched)
            for country in country_names:   #Creation of one column per country
                country_cols = [col for col in df.columns if col.startswith(country)]  
                if country_cols:
                    # Separate up/down if present, then sum abs for total magnitude
                    up_cols = [c for c in country_cols if f"{country}__Upward" in c]
                    down_cols = [c for c in country_cols if f"{country}__Downward" in c]
                    total_col = f"{country}_total_activation_mwh" #creation of column name
                   
                    if up_cols and down_cols: #if both directions exist, finds absolute total activation
                        df[total_col] = df[up_cols].abs().sum(axis=1, skipna=True) + df[down_cols].abs().sum(axis=1, skipna=True)
                    else:
                        # If no direction split, sum all
                        df[total_col] = df[country_cols].abs().sum(axis=1, skipna=True)

            # Add combined column: Sum of per-country total activations (Baltic total magnitude)
            per_country_cols = [f"{country}_total_activation_mwh" for country in country_names if f"{country}_total_activation_mwh" in df.columns]
            if len(per_country_cols) > 0:
                combined_col = "baltic_total_activation_mwh"
                df[combined_col] = df[per_country_cols].sum(axis=1, skipna=True)

        elif data_id == 'imbalance_volumes_v2':
            # For each country, select its single imbalance column
            for country in country_names:
                country_cols = [col for col in df.columns if col.startswith(country) and ('MWh' in col or country in col)]
                if country_cols:
                    if len(country_cols) == 1:
                        imb_col = f"{country}_imbalance_mwh"
                        df[imb_col] = df[country_cols[0]]
            
            # Add combined column: Sum of per-country net imbalances (Baltic total)
            per_country_cols = [f"{country}_imbalance_mwh" for country in country_names if f"{country}_imbalance_mwh" in df.columns]
            if len(per_country_cols) > 0:
                combined_col = "baltic_net_imbalance_mwh"
                df[combined_col] = df[per_country_cols].sum(axis=1, skipna=True)
        return df
    
    #Handling exceptions and errors to return empty dataframe 
    except Exception as e:
        return pd.DataFrame()
        
def compute_basic_metrics(acts_df, imb_df):
    metrics = {}    #creating a dictionary for calculated metrics
    country_names = ['Estonia', 'Latvia', 'Lithuania']
    
    # Per-region activation metrics
    if not acts_df.empty:
        for country in country_names:
            col = f"{country}_total_activation_mwh"
            if col in acts_df.columns:
                activations = acts_df[col].dropna()  # Already absolute magnitude
                if not activations.empty:
                    metrics[f"{country} Avg Activation (MWh)"] = round(activations.mean(), 2)   # average activation values
                    metrics[f"{country} Max Activation (MWh)"] = round(activations.max(), 2)    # maximum activation values
                    metrics[f"{country} Total Dispatched (MWh over period)"] = round(activations.sum(), 2) # total dispatched values
        
        # Overall activation metrics (using combined Baltic column if available, else return empty)
        combined_act_col = "baltic_total_activation_mwh"
        if combined_act_col in acts_df.columns:
            overall_acts = acts_df[combined_act_col].dropna()
        else:
            overall_acts = pd.Series(dtype=float)  # Empty
        if not overall_acts.empty:
            metrics['Overall Avg Activation (MWh)'] = round(overall_acts.mean(), 2)
            metrics['Overall Max Activation (MWh)'] = round(overall_acts.max(), 2)
            metrics['Overall Total Dispatched (MWh)'] = round(overall_acts.sum(), 2)
    
    # Per-region imbalance metrics (using signed net values)
    if not imb_df.empty:
        for country in country_names:
            col = f"{country}_imbalance_mwh"
            if col in imb_df.columns:
                volumes = imb_df[col].dropna()  # Signed (positive=surplus, negative=deficit)
                if not volumes.empty:
                    metrics[f"{country} Avg Imbalance (MWh)"] = round(volumes.mean(), 2)
                    metrics[f"{country} Total Net Imbalance (MWh)"] = round(volumes.sum(), 2)
                    # Also add magnitude for insight (abs avg and total)
                    abs_volumes = volumes.abs()
                    metrics[f"{country} Avg |Imbalance| (MWh)"] = round(abs_volumes.mean(), 2)
                    metrics[f"{country} Total |Imbalance| (MWh)"] = round(abs_volumes.sum(), 2)
                    metrics[f"{country} Non-Null Intervals"] = len(volumes)
        
        # Overall imbalance metrics (using combined Baltic column if available, else return empty)
        combined_imb_col = "baltic_net_imbalance_mwh"
        if combined_imb_col in imb_df.columns:
            overall_volumes = imb_df[combined_imb_col].dropna()
        else:
            overall_volumes = pd.Series(dtype=float)  # Empty
        if not overall_volumes.empty:
            metrics['Overall Avg Imbalance (MWh)'] = round(overall_volumes.mean(), 2)
            metrics['Overall Total Net Imbalance (MWh)'] = round(overall_volumes.sum(), 2)
            # Magnitude for overall
            abs_overall = overall_volumes.abs()
            metrics['Overall Avg |Imbalance| (MWh)'] = round(abs_overall.mean(), 2)
            metrics['Overall Total |Imbalance| (MWh)'] = round(abs_overall.sum(), 2)
    
    # Overall correlation (total activation vs. total |imbalance| across regions, using combined or summed)
    combined_act_col = "baltic_total_activation_mwh"
    combined_imb_col = "baltic_net_imbalance_mwh"
    if (combined_act_col in acts_df.columns or any(f"{c}_total_activation_mwh" in acts_df.columns for c in country_names)) and \
       (combined_imb_col in imb_df.columns or any(f"{c}_imbalance_mwh" in imb_df.columns for c in country_names)):
        # Use combined if available
        if combined_act_col in acts_df.columns:
            total_acts = acts_df[combined_act_col]
        
        if combined_imb_col in imb_df.columns:
            total_imbs = imb_df[combined_imb_col]
        
        # create two columns with total activation and absolute total imbalance
        combined = pd.merge(total_acts, total_imbs.abs(), left_index=True, right_index=True, how='inner').dropna()
        if not combined.empty:
            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1]) # find if linear relationship between total activation and abs. total imbalance exists.
            metrics['Overall Corr (Total Activation vs. Total |Imbalance|)'] = round(corr, 3) if not pd.isna(corr) else 'N/A'
        else:
            metrics['Overall Corr (Total Activation vs. Total |Imbalance|)'] = 'N/A (No overlapping data)'
        
        # aFRR activation proportions to absolute imbalances
    for country in country_names:
        act_col = f"{country}_total_activation_mwh"
        imb_col = f"{country}_imbalance_mwh"
        if act_col in acts_df.columns and imb_col in imb_df.columns:
            total_activation = acts_df[act_col].sum()
            total_abs_imbalance = imb_df[imb_col].abs().sum()
            metrics[f"{country} Activation/|Imbalance|"] = round(total_activation / total_abs_imbalance, 3) if total_abs_imbalance > 0 else 'N/A'
        else:
            metrics[f"{country} Activation/|Imbalance|"] = 'N/A'
    
    # Overall Baltic proportion
    if combined_act_col in acts_df.columns and combined_imb_col in imb_df.columns:
        total_activation = acts_df[combined_act_col].sum()
        total_abs_imbalance = imb_df[combined_imb_col].abs().sum()
        metrics["Baltic Activation/|Imbalance|"] = round(total_activation / total_abs_imbalance, 3) if total_abs_imbalance > 0 else 'N/A'
    else:
        metrics["Baltic Activation/|Imbalance|"] = 'N/A'    
    
    return metrics

    # Plotting data 
def plot_data(acts_df, imb_df, metrics):
    # Separate lines per region + combined Baltic for aFRR activations and imbalances
    fig, ax1 = plt.subplots(figsize=(14, 7))  # Wider for multiple lines
    country_names = ['Estonia', 'Latvia', 'Lithuania']
    colors = {
        'Estonia': 'blue',
        'Latvia': 'green',
        'Lithuania': 'orange',
        'Baltic': 'purple'
    }
    # Linestyles
    ls_activation = '-'   # solid for activations
    ls_imbalance = '--'   # dashed for imbalances
    lines, labels = [], []
    # Plot activations (left axis)
    for country in country_names:
            col = f"{country}_total_activation_mwh"
            if col in acts_df.columns:
                line, = ax1.plot(acts_df.index, acts_df[col], 
                                 color=colors[country], linestyle=ls_activation,
                                 linewidth=2, marker='o', markersize=4,
                                 label=f"{country} Activation")
                lines.append(line)
                labels.append(f"{country} Activation")        
    combined_act_col = "baltic_total_activation_mwh"
    
    if combined_act_col in acts_df.columns:
            line, = ax1.plot(acts_df.index, acts_df[combined_act_col],
                             color=colors['Baltic'], linestyle=ls_activation,
                             linewidth=3, marker='*', markersize=6,
                             label="Baltic Activation")
            lines.append(line)
            labels.append("Baltic Activation")
        
            ax1.set_ylabel('Activation (MWh)')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_ylim(bottom=0)
    # Plot imbalances (right axis)
            ax2 = ax1.twinx()   # Overlaying second y-axis
    for country in country_names:
            col = f"{country}_imbalance_mwh"
            if col in imb_df.columns:
                line, = ax2.plot(imb_df.index, imb_df[col],
                                 color=colors[country], linestyle=ls_imbalance,
                                 linewidth=2, marker='s', markersize=4,
                                 label=f"{country} Imbalance")
                lines.append(line)
                labels.append(f"{country} Imbalance")
    # Plot Baltic combined imbalances    
            combined_imb_col = "baltic_net_imbalance_mwh"
    if combined_imb_col in imb_df.columns:
            line, = ax2.plot(imb_df.index, imb_df[combined_imb_col],
                             color=colors['Baltic'], linestyle=ls_imbalance,
                             linewidth=3, marker='D', markersize=6,
                             label="Baltic Imbalance")
            lines.append(line)
            labels.append("Baltic Imbalance")
        
            ax2.set_ylabel('Imbalance (MWh)')
            ax2.tick_params(axis='y', labelcolor='red')

    # Formatting title, legend
    plt.title(f'aFRR Activations vs. Imbalance Volumes (Estonia, Latvia, Lithuania, Baltic)\n15-min Resolution ({PARAMS["start_date"]} to {PARAMS["end_date"]} EET)')
    plt.xlabel('Hour of Day (EET)')

    ax1.legend(lines, labels, loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Format x-axis as hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig('afrr_activations_imbalance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Fetching aFRR Activations...")
    acts_df = fetch_data('activations_afrr')
    if not acts_df.empty:
        print(f"Saved parsed aFRR activations data ({acts_df.shape[0]} 15-min rows).")

    print("\nFetching Imbalance Volumes v2...")
    imb_df = fetch_data('imbalance_volumes_v2')
    if not imb_df.empty:
        print(f"Saved parsed imbalance data ({imb_df.shape[0]} 15-min rows).")

    metrics = compute_basic_metrics(acts_df, imb_df)
    print("\n=== Basic Metrics (Per-Region + Baltic: aFRR Activations vs. Signed Imbalances) ===")
    if metrics:
        for key, value in sorted(metrics.items()):
            print(f"{key}: {value}")

    plot_data(acts_df, imb_df, metrics)
