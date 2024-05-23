# TwitterScraper

#### This repository contains two main files for conducting sentiment and emotion analysis on Twitter data using machine learning models. The train_twitter.py file is used for uploading training data and training three different models: Linear SVC, Naive Bayes, and BERT (both bert-base-uncased and bert-large-uncased). The trained models are saved and can be loaded for testing accuracy. 

#### The twitter.py file is used for retrieving tweets from the Twitter API based on a keyword, specifying the count of tweets, selecting the top percent of emotions, and choosing the model of choice for sentiment and empath analysis using the saved models. The user interface for interacting with these functionalities is built using Streamlit.

## Files

#### train_twitter.py: Python script for uploading training data and training machine learning models (Linear SVC, Naive Bayes, BERT).
#### twitter.py: Python script for retrieving tweets from Twitter API based on a keyword, specifying the count of tweets, selecting the top percent of emotions.


