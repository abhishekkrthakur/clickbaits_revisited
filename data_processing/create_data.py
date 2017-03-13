# coding: utf-8
"""
Create usable data after scraping public facebook pages
@author: Abhishek Thakur
"""

import pandas as pd

buzzfeed = pd.read_csv('../data/buzzfeed_facebook_statuses.csv',
                       usecols=['link_name', 'status_type', 'status_link'])

clickhole = pd.read_csv('../data/clickhole_facebook_statuses.csv',
                        usecols=['link_name', 'status_type', 'status_link'])

cnn = pd.read_csv('../data/cnn_facebook_statuses.csv',
                  usecols=['link_name', 'status_type', 'status_link'])

nytimes = pd.read_csv('../data/nytimes_facebook_statuses.csv',
                      usecols=['link_name', 'status_type', 'status_link'])

stopclickbait = pd.read_csv('../data/StopClickBaitOfficial_facebook_statuses.csv',
                            usecols=['link_name', 'status_type', 'status_link'])

upworthy = pd.read_csv('../data/Upworthy_facebook_statuses.csv',
                       usecols=['link_name', 'status_type', 'status_link'])

wikinews = pd.read_csv('../data/wikinews_facebook_statuses.csv',
                       usecols=['link_name', 'status_type', 'status_link'])

wikinews.link_name = wikinews.link_name.apply(lambda x: str(x).replace(' - Wikinews, the free news source', ''))
buzzfeed = buzzfeed[buzzfeed.status_type == 'link']
clickhole = clickhole[clickhole.status_type == 'link']
cnn = cnn[cnn.status_type == 'link']
nytimes = nytimes[nytimes.status_type == 'link']
stopclickbait = stopclickbait[stopclickbait.status_type == 'link']
upworthy = upworthy[upworthy.status_type == 'link']
wikinews = wikinews[wikinews.status_type == 'link']

cnn = cnn.sample(frac=1).head(10000)
nytimes = nytimes.sample(frac=1).head(13000)

clickbaits = pd.concat([buzzfeed, clickhole, stopclickbait, upworthy])
non_clickbaits = pd.concat([cnn, nytimes, wikinews])

clickbaits.to_csv('../data/clickbaits.csv', index=False)
non_clickbaits.to_csv('../data/non_clickbaits.csv', index=False)
