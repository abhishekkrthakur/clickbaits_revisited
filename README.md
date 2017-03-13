# Clickbaits Revisited

This repository provides the code used for : https://www.linkedin.com/pulse/clickbaits-revisited-deep-learning-title-content-features-thakur


### Data Collection
To run the code you must first collect the data:

- Get facebook page parser from: https://github.com/minimaxir/facebook-page-post-scraper
- Run the python script: get_fb_posts_fb_page.py for buzzfeed, upworthy, cnn, nytimes, wikinews, clickhole and StopClickBaitOfficial
- Save all the CSVs obtained from above step in data/


### Data Pre-Processing
After the data has been collected, you need to run the following files to obtain training and test data. The order is important!

    - $ cd data_processing
    - $ python create_data.py
    - $ python html_scraper.py
    - $ python feature_generation.py
    - $ python merge_data.py
    - $ python data_cleaning.py
 
After the steps above, you will end up with train.csv and test.csv in data/

Please note that the above steps will require a lot of memory. So, if you have anything less than 64GB, please modify the code according to your needs.

### GloVe embeddings

Obtain GloVe embeddings from the following URL:

    http://nlp.stanford.edu/data/glove.840B.300d.zip
    
Extract the zip and place the CSV in data/


### Deepnets

After all the above steps, you are ready to go and play around with the deep neural networks to classify clickbaits

Change directory to deepnets/
    
    cd deepnets/
     
The deepnets are as folllows:
    
    LSTM_Title.py : LSTM on title text without GloVe embeddings
    LSTM_Title_Content.py : LSTM on title text and content text without GloVe embeddings
    LSTM_Title_Content_with_GloVe.py : LSTM on title and content text with GloVe emebeddings
    TDD_Title_Content_with_Glove.py : Time distributed dense on title and content text with GloVe embeddings
    LSTM_Title_Content_Numerical_with_GloVe.py : LSTM on title + content text with GloVe embeddings & dense net for numerical features.
     

### Performance

The network with LSTM on title and content text with GloVe embeddings with numerical features achieves an accuracy of 0.996 during validation and 0.992 on the test set.

All models were trained on NVIDIA TitanX, Ubuntu 16.04 system with 64GB memory.

