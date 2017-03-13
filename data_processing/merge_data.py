# coding: utf-8
"""
Merge original clickbait CSVs with features
@author: Abhishek Thakur
"""
import pandas as pd

clickbait_titles = pd.read_csv('../data/clickbaits.csv')
non_clickbait_titles = pd.read_csv('../data/non_clickbaits.csv')

clickbait_features = pd.read_csv('../data/clickbait_website_features.csv')
non_clickbait_features = pd.read_csv('../data/non_clickbait_website_features.csv')

clickbait_full = pd.concat([clickbait_titles, clickbait_features], axis=1)
non_clickbait_full = pd.concat([non_clickbait_titles, non_clickbait_features], axis=1)

clickbait_full['label'] = 1
non_clickbait_full['label'] = 0

fulldata = pd.concat([clickbait_full, non_clickbait_full])
fulldata = fulldata.sample(frac=1).reset_index(drop=True)
fulldata = fulldata[fulldata.html_len != -1]

fulldata.to_csv('../data/fulldata.csv', index=False)
