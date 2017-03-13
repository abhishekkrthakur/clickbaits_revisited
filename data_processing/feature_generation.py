# coding: utf-8
"""
Generate numerical content based and text features from HTMLs
@author: Abhishek Thakur
"""

import pandas as pd
import cPickle
from bs4 import BeautifulSoup
from goose import Goose
from collections import Counter
import string
from joblib import Parallel, delayed
import sys
from tqdm import tqdm

stop_domains = ['buzzfeed', 'clickhole', 'cnn', 'wikinews', 'upworthy', 'nytimes']


def features(html):
    try:
        soup = BeautifulSoup(html, "lxml")
        g = Goose()
        try:
            goose_article = g.extract(raw_html=html)
        except TypeError:
            goose_article = None
        except IndexError:
            goose_article = None

        size = sys.getsizeof(html)
        html_len = len(html)
        number_of_links = len(soup.find_all('a'))
        number_of_buttons = len(soup.find_all('button'))
        number_of_inputs = len(soup.find_all('input'))
        number_of_ul = len(soup.find_all('ul'))
        number_of_ol = len(soup.find_all('ol'))
        number_of_lists = number_of_ol + number_of_ul
        number_of_h1 = len(soup.find_all('h1'))
        number_of_h2 = len(soup.find_all('h2'))
        if number_of_h1 > 0:
            h1_len = 0
            h1_text = ''
            for x in soup.find_all('h1'):
                text = x.get_text().strip()
                h1_text += text + ' '
                h1_len += len(text)
            total_h1_len = h1_len
            avg_h1_len = h1_len * 1. / number_of_h1
        else:
            total_h1_len = 0
            avg_h1_len = 0
            h1_text = ''

        if number_of_h2 > 0:
            h2_len = 0
            h2_text = ''
            for x in soup.find_all('h2'):
                text = x.get_text().strip()
                h2_len += len(text)
                h2_text += text + ' '
            total_h2_len = h2_len
            avg_h2_len = h2_len * 1. / number_of_h2
        else:
            total_h2_len = 0
            avg_h2_len = 0
            h2_text = ''
        if goose_article is not None:
            textdata = goose_article.meta_description + ' ' + h1_text + ' ' + h2_text
            textdata = "".join(l for l in textdata if l not in string.punctuation)
            textdata = textdata.strip().lower().split()
            textdata = [word for word in textdata if word.lower() not in stop_domains]
            textdata = ' '.join(textdata)
        else:
            textdata = h1_text + ' ' + h2_text
            textdata = "".join(l for l in textdata if l not in string.punctuation)
            textdata = textdata.strip().lower().split()
            textdata = [word for word in textdata if word.lower() not in stop_domains]
            textdata = ' '.join(textdata)

        number_of_images = len(soup.find_all('img'))

        number_of_tags = len([x.name for x in soup.find_all()])
        number_of_unique_tags = len(Counter([x.name for x in soup.find_all()]))

        return [size, html_len, number_of_links, number_of_buttons,
                number_of_inputs, number_of_ul, number_of_ol, number_of_lists,
                number_of_h1, number_of_h2, total_h1_len, total_h2_len, avg_h1_len, avg_h2_len,
                number_of_images, number_of_tags, number_of_unique_tags,
                textdata]
    except:
        return [-1, -1, -1, -1,
                -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1,
                -1, -1, -1,
                "no data"]


clickbait_html = cPickle.load(open('../data/clickbait_html.pkl'))
clickbait_features = Parallel(n_jobs=50)(delayed(features)(html) for html in tqdm(clickbait_html))

clickbait_features_df = pd.DataFrame(clickbait_features,
                                     columns=["size", "html_len", "number_of_links", "number_of_buttons",
                                              "number_of_inputs", "number_of_ul", "number_of_ol", "number_of_lists",
                                              "number_of_h1", "number_of_h2", "total_h1_len", "total_h2_len",
                                              "avg_h1_len", "avg_h2_len",
                                              "number_of_images", "number_of_tags", "number_of_unique_tags",
                                              "textdata"])

clickbait_features_df.to_csv('../data/clickbait_website_features.csv', index=False, encoding='utf-8')

non_clickbait_html = cPickle.load(open('../data/non_clickbait_html.pkl'))
non_clickbait_features = Parallel(n_jobs=50)(delayed(features)(html) for html in tqdm(non_clickbait_html))

non_clickbait_features_df = pd.DataFrame(non_clickbait_features,
                                         columns=["size", "html_len", "number_of_links", "number_of_buttons",
                                                  "number_of_inputs", "number_of_ul", "number_of_ol", "number_of_lists",
                                                  "number_of_h1", "number_of_h2", "total_h1_len", "total_h2_len",
                                                  "avg_h1_len", "avg_h2_len",
                                                  "number_of_images", "number_of_tags", "number_of_unique_tags",
                                                  "textdata"])

non_clickbait_features_df.to_csv('../data/non_clickbait_website_features.csv', index=False, encoding='utf-8')
