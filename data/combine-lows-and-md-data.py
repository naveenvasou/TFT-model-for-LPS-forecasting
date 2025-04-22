import numpy as np
import pandas as pd

lows_data = pd.read_csv("processed/Lows_track_with_variable_data.csv")
lows_data['Genesis_Date'] = pd.to_datetime(lows_data['Genesis_Date'], format='%d-%m-%Y %H:%M')
lows_data['Genesis_Date'] = lows_data['Genesis_Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
lows_data['DateTime'] = pd.to_datetime(lows_data['DateTime'], format='%d-%m-%Y %H:%M')
lows_data['DateTime'] = lows_data['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')

md_data = pd.read_csv("C:/Users/Naveen Kumar/Downloads/major/MDs_track_with_variable_data.csv")
md_data['Genesis_Date'] = pd.to_datetime(md_data['Genesis_Date'])
md_data['Genesis_Date'] = md_data['Genesis_Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
md_data['DateTime'] = pd.to_datetime(md_data['DateTime'])
md_data['DateTime'] = md_data['DateTime'].dt.strftime('%Y-%m-%d %H:%M:%S')


combined_df = pd.concat([lows_data, md_data])
combined_df = combined_df.sort_values(by=['Genesis_Date'])
combined_df = combined_df.reset_index(drop=True)

id = { pd.unique(combined_df["Genesis_Date"])[i]: i+1 for i in range(len(pd.unique(combined_df["Genesis_Date"])))}
combined_df["id"] = [id[combined_df["Genesis_Date"][i]] for i in range(len(combined_df))]

combined_df["month"] = pd.to_datetime(combined_df["DateTime"]).dt.month
combined_df["day"] = pd.to_datetime(combined_df["DateTime"]).dt.day
combined_df["hour"] = pd.to_datetime(combined_df["DateTime"]).dt.hour

combined_df.to_csv("processed/all-lps-dataframe.csv")
