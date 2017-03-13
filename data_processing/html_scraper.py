# coding: utf-8
"""
Scrape and save html for all links in clickbait and non_clickbait CSVs
@author: Abhishek Thakur
"""
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

import pandas as pd
import requests
from joblib import Parallel, delayed
import cPickle
from tqdm import tqdm


def html_extractor(url):
    try:
        cookies = dict(cookies_are='working')
        r = requests.get(url, cookies=cookies)
        return r.text
    except:
        return "no html"


clickbaits = pd.read_csv('../data/clickbaits.csv')
non_clickbaits = pd.read_csv('../data/non_clickbaits.csv')

clickbait_urls = clickbaits.status_link.values
non_clickbait_urls = non_clickbaits.status_link.values


clickbait_html = Parallel(n_jobs=20)(delayed(html_extractor)(u) for u in tqdm(clickbait_urls))
cPickle.dump(clickbait_html, open('../data/clickbait_html.pkl', 'wb'), -1)

non_clickbait_html = Parallel(n_jobs=20)(delayed(html_extractor)(u) for u in tqdm(non_clickbait_urls))
cPickle.dump(non_clickbait_html, open('../data/non_clickbait_html.pkl', 'wb'), -1)
