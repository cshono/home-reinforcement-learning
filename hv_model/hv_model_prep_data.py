# -*- coding: utf-8 -*-
"""
Takes the CSVs from multiple sources
Converts to single time resolution (1 minute)
Include following variables:
    Timestamp
    hvac_w
    csp_0
    hsp_0
    csp_1
    hsp_1
    csp_2
    hsp_2
    rmt_0
    rmt_1
    rmt_2
    ts_mode_0
    ts_mode_1
    ts_mode_2
    outdoor_temp
"""

# Import Libraries
import pandas as pd
import pandas.io.common
import numpy as np
from statistics import mode, StatisticsError 
from pytemp import pytemp 
import os
import re

# Month, Year GLOBALS 
YEAR = "2019"
MONTH = "09" 
DAYS_IN_MONTH = 30 

# Define Inputs
dir_tstat = "../../data-raw/iot_data/{}-{}/".format(YEAR[2:], MONTH)
dir_load = "../../data-raw/preprocessed_data/premise/{}-{}/".format(YEAR, MONTH)
dir_weather = "../../data-raw/weather_data/"
dir_out = "../../data/hv_model/" 

#rl_list = os.listdir(dir_tstat)[0:5] 
rl_list = [d for d in os.listdir(dir_tstat) if "." not in d] 
#rl_list = ["RL1", "RL3"] 
#rl_list = ["RL29"] 


FILE_TSTAT = "hvac_"
FILE_LOAD = "{}_{}-{}.csv"
FILE_WEATHER = "weather_{}-{}.csv".format(YEAR[2:], MONTH) 
FILE_OUT = "{}_hv_{}-{}.P" 

start_time = "{}-{}-01".format(YEAR, MONTH) 
end_time = "{}-{}-{}".format(YEAR, MONTH, DAYS_IN_MONTH) 
tstat_col_substrings = "Timestamp|Set|Activity|Room Temp"
load_col_substrings = "Timestamp|RLID|value" 
weather_col_substrings = "Timestamp|TEMP" 
time_res = "1H" 

tstat_col_dict = {
        "Zone0 Cool Setpoint": "csp_0",
        "Zone1 Cool Setpoint": "csp_1",
        "Zone2 Cool Setpoint": "csp_2",

        "Zone0 Heat Setpoint": "hsp_0",
        "Zone1 Heat Setpoint": "hsp_1",
        "Zone2 Heat Setpoint": "hsp_2",

        "Zone0 Room Temp": "rmt_0",
        "Zone1 Room Temp": "rmt_1",
        "Zone2 Room Temp": "rmt_2",

        "Zone0 Current Activity": "ts_mode_0",
        "Zone1 Current Activity": "ts_mode_1",
        "Zone2 Current Activity": "ts_mode_2"
}



load_col_dict = {"value": "hvac_W"} 

weather_col_dict = {"AMBIENT TEMP": "amb_temp"} 


def clean_tstat_raw(df, start_time, end_time, col_substrings, rlid):
    # Set up timestamps and convert to local time
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC") # Add UTC timezone
    df["Timestamp"] = df["Timestamp"].dt.tz_convert("US/Central").dt.tz_localize(None) # Convert to Central, then drop tz.
    df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)]


    if df.shape[0] == 0:
        print("{}: No datapoints between {} and {}!!!".format(rlid, start_time, end_time))
    df = df.filter(regex = col_substrings)

    # Remove null columns
    df = df.dropna(axis=1, how='all')

    # Remove the ".1" from the end of column names
    df = df.rename(columns = lambda x: re.sub('\.1', '', x))

    df["RLID"] = rlid

    return df

def clean_load_raw(df, start_time, end_time, col_substrings):
    df["Timestamp"] = pd.to_datetime(df["timestamp.ct"]) 
    df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)]
    df = df[df["beopt.group"] == "Heating/Cooling"]
    df = df.filter(regex = col_substrings) 
    
    return df 

def clean_weather_raw(df, start_time, end_time, col_substrings): 
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]) 
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC") # Add UTC timezone
    df["Timestamp"] = df["Timestamp"].dt.tz_convert("US/Central").dt.tz_localize(None) # Convert to Central, then drop tz.
    df =  df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)]
    
    df = df.filter(regex = col_substrings) 
    
    df["AMBIENT TEMP"] = df["AMBIENT TEMP"].apply(lambda t: pytemp(t, "c", "f")) 
    
    return df 


def resample_tstat_15min(df):
    df = df.set_index("Timestamp")
    df = df.resample("min").ffill()
    df = df.fillna(method="ffill").fillna(method="bfill")
    df = df.resample(time_res).apply(lambda x: mode(x))
    df = df.reset_index()
    return df

def resample_load_15min(df):
    df = df.set_index("Timestamp")
    df = df.resample(time_res).mean() 
    df = df.reset_index() 
    return df 

def resample_weather_15min(df):
    df = df.set_index("Timestamp")
    df = df.resample(time_res).mean() 
    df = df.reset_index() 
    return df 

def convert_col_names(df, col_dict):
    df = df.rename(columns = col_dict)
    return df

def merge_datasets(df1, df2, df3): 
    df1 = df1.set_index("Timestamp") 
    df2 = df2.set_index("Timestamp") 
    df3 = df3.set_index("Timestamp") 
    
    df_out = pd.concat([df1, df2, df3], axis=1) 
    df_out = df_out.reset_index() 
    return df_out 

def calc_degree_days(df, zone): 
    h_col = "hsp_{}".format(zone) 
    c_col = "csp_{}".format(zone) 
    if h_col not in df or c_col not in df: 
        print("Either {} or {} not in dataset".format(h_col, c_col)) 
        return None
    
    df["cdd_{}".format(zone)] = df["amb_temp"] - df[c_col] 
    df["hdd_{}".format(zone)] = df[h_col] - df["amb_temp"] 
    
    df["cdd_{}".format(zone)] = df["cdd_{}".format(zone)].apply(lambda x: max(x, 0)) 
    df["hdd_{}".format(zone)] = df["hdd_{}".format(zone)].apply(lambda x: max(x, 0))
    
    return df

def try_mode(x): 
    try: 
        m = mode(x) 
    except StatisticsError: 
        if len(x) > 1: 
            m = x[0] 
        else: 
            m = x
    return m 

def resample_hvac_to_daily(df): 
    """
    cols_apply_avg = {
        "csp_0": "mean",
        "hsp_0": "mean",
        "csp_1": "mean",
        "hsp_1": "mean",
        "csp_2": "mean",
        "hsp_2": "mean",
        "rmt_0": "mean",
        "rmt_1": "mean",
        "rmt_2": "mean",
        "amb_temp": "mean", 
    }
    
    cols_apply_sum = {
        "hvac_W": "sum"
    }
    
    cols_apply_first = {
        "RLID": "first"
    }
    
    cols_apply_mode = {
        "ts_mode_0": lambda x: try_mode(x), 
        "ts_mode_1": lambda x: try_mode(x), 
        "ts_mode_2": lambda x: try_mode(x), 
    }
    """
    
    cols_apply_avg = { 
        "hdd_0": "mean", 
        "hdd_1": "mean", 
        "hdd_2": "mean", 
        "cdd_0": "mean", 
        "cdd_1": "mean", 
        "cdd_2": "mean", 
    }
    
    cols_apply_sum = {
        "hvac_W": "sum" 
    }
    
    cols_apply_first = {
        "RLID": "first"
    }

    # Filter dictionary by keeping elements whose keys are divisible by 2
    #newDict = { key:value for (key,value) in dictOfNames.items() if key % 2 == 0}
    
    cols_apply_avg = {key:value for (key,value) in cols_apply_avg.items() if key in df} 
    cols_apply_sum = {key:value for (key,value) in cols_apply_sum.items() if key in df} 
    cols_apply_first = {key:value for (key,value) in cols_apply_first.items() if key in df} 
    #cols_apply_mode = {key:value for (key,value) in cols_apply_mode.items() if key in df} 
    #cols_apply_mode_err = {key:value for (key,value) in cols_apply_mode_err.items() if key in df} 
    
    df = df.set_index("Timestamp") 
    df_means = df.resample("D").agg(cols_apply_avg) 
    df_sums = df.resample("D").agg(cols_apply_sum) 
    df_firsts = df.resample("D").agg(cols_apply_first) 
    #df_modes = df.resample("D").agg(cols_apply_mode) 

    # Rename hvac_W to hvac_Wh 
    df_sums["hvac_kWh"] = df_sums["hvac_W"]/1000 
    del df_sums["hvac_W"] 
    
    #df_out = pd.concat([df_means, df_sums, df_firsts, df_modes], axis=1) 
    df_out = pd.concat([df_means, df_sums, df_firsts], axis=1) 
    
    df_out = df_out.dropna() 
    return df_out     

def import_tstat(dir_in): 
    
    if os.listdir(dir_in) == []: 
        return None 
    
    tstat_file = [f for f in os.listdir(dir_in) if f.startswith(FILE_TSTAT)][0]
    tstat_filepath = os.path.join(dir_in, tstat_file)
    
    try:
        tstat = pd.read_csv(tstat_filepath)
    except pandas.io.common.EmptyDataError:
        return None
    
    tstat = clean_tstat_raw(tstat, start_time, end_time, tstat_col_substrings, rlid)
    if tstat.shape[0] == 0: 
        return None
    
    tstat = resample_tstat_15min(tstat)
    tstat = convert_col_names(tstat, tstat_col_dict)
    
    return tstat 

def import_load(filepath): 
    try: 
        load = pd.read_csv(filepath)
    except FileNotFoundError: 
        print("Could not find load file: {}".format(rlid)) 
        return None 
    
    load = clean_load_raw(load, start_time, end_time, load_col_substrings) 
    load = resample_load_15min(load) 
    load = convert_col_names(load, load_col_dict) 
    
    return load 

def import_weather(filepath): 
    weather = pd.read_csv(filepath) 
    weather = clean_weather_raw(weather, start_time, end_time, weather_col_substrings) 
    weather = resample_weather_15min(weather) 
    weather = convert_col_names(weather, weather_col_dict) 
    return weather 


# Import and Merge HVAC Data
hvac = pd.DataFrame()
for rlid in rl_list:
    print("Processing: {}".format(rlid)) 
    

    # Import TSTAT data and clean
    tstat = import_tstat(os.path.join(dir_tstat, rlid)) 
    if type(tstat) == type(None): 
        print("No thermostat data...") 
        continue 

    # Import Load data and clean
    load = import_load(os.path.join(dir_load, FILE_LOAD.format(rlid, YEAR, MONTH))) 
    if type(load) == type(None): 
        print("No load data...") 
        continue 
    
    weather = import_weather(os.path.join(dir_weather, FILE_WEATHER)) 
    if type(weather) == type(None): 
        print("No weather data...") 
        continue 
    

    
    # combine all df's based on timestamp 
    tmp_hvac = merge_datasets(tstat, load, weather) 
    tmp_hvac = calc_degree_days(tmp_hvac, 0) 
    if type(tmp_hvac) == type(None): 
        continue 
    tmp_hvac = resample_hvac_to_daily(tmp_hvac) 
    
    #tmp_hvac.to_pickle(os.path.join(dir_out, FILE_OUT.format(rlid, YEAR, MONTH))) 
    #tmp_hvac.to_csv("{}_hv_{}-{}.csv".format(rlid, YEAR, MONTH)) 
    
    # Add tstat to full dataframe
    hvac = hvac.append(tmp_hvac) 
    
hvac.to_csv("hvac_daily_{}-{}.csv".format(YEAR, MONTH)) 




