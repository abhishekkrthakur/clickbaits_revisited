# coding: utf-8
"""
Cleans the data more and separates into training and test sets
@author: Abhishek Thakur
"""

import pandas as pd
from sklearn.cross_validation import train_test_split

internet_stop_words = ['site', 'navigation', 'new', 'times', 'york', 'information', 'index',
                       'like', 'related', 'search', 'follow', 'subscribe', 'subscribed', 'subscribing',
                       'spam', 'twitter', 'pinterest', 'facebook', 'google', 'privacy', 'policy', 'feedback',
                       'tweet', 'tweets', 'disclaimer', 'buzzfeed', 'clickhole', 'upworthy', 'cnn', 'nytimes',
                       'wikinews', 'instagram', 'newsletter', 'copyright', 'cnn.com', 'nytimes.com',
                       'buzzfeed.com', 'upworthy.com', 'clickhole.com', 'wikinews.com']


def remove_internet_stop_words(x):
    return ' '.join([word for word in str(x).lower().split() if word not in internet_stop_words])


df = pd.read_csv('../data/fulldata.csv')
df = df.drop_duplicates()

df.textdata = df.textdata.apply(lambda x: str(x).replace('report an issue thanks', '').strip())

df.textdata = df.textdata.apply(remove_internet_stop_words)
df.link_name = df.link_name.apply(remove_internet_stop_words)

df = df.drop(['status_type', 'status_link'], axis=1)

train_df, test_df = train_test_split(df, stratify=df.label.values, random_state=42, test_size=0.1)

train_df.to_csv('../data/train.csv', index=False, encoding='utf-8')
test_df.to_csv('../data/test.csv', index=False, encoding='utf-8')
