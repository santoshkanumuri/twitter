import streamlit as st
import tweepy
from googletrans import Translator  # Assuming you're using the googletrans library
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from itertools import combinations
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from train_twitter import evaluate_saved_bert_model
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import requests
from train_twitter import label_encode_df


# Function to retrieve tweets
def retrieve_tweets(bearer_token, search_str, num_tweets):
    search_str=search_str+" -is:retweet"
    keyword=search_str
    total_tweets=num_tweets
    max_results_per_request = 100
    tweets = []

    # Define the Twitter API endpoint for searching tweets with pagination
    endpoint = f"https://api.twitter.com/2/tweets/search/recent?query={keyword}&max_results={max_results_per_request}"

    # Set up the authorization header with the bearer token
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }

    # Function to fetch tweets
    def fetch_tweets(url):
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    # Fetch tweets with pagination
    next_token = None
    while len(tweets) < total_tweets:
        if next_token:
            url = f"{endpoint}&next_token={next_token}"
        else:
            url = endpoint

        data = fetch_tweets(url)
        new_tweets = data.get('data', [])
        tweets.extend(new_tweets)

        # Break if there are no more tweets to fetch
        if 'next_token' not in data.get('meta', {}):
            break
        next_token = data['meta']['next_token']

        # Break if we have reached the total number of desired tweets
        if len(tweets) >= total_tweets:
            break

    # Trim the list to the desired number of tweets
    tweets = tweets[:total_tweets]
    cleaned_tweets = []
    for tweet in tweets:
        t = tweet['text']
        t = re.sub(r"(?:\@|https?\://|\#)\S+", "", t)
        print(t)
        cleaned_tweets.append(t)
    print(cleaned_tweets)
    return cleaned_tweets


# Function to translate tweets
def translate_tweets(tweets):
    translator = Translator()
    translated_tweets = []
    for tweet in tweets:
        try:
            translation = translator.translate(tweet, dest='en')
            translated_tweets.append(translation.text)
        except AttributeError as e:
            print("attribute error"+tweet)
        except TypeError as e:
            print("type error" + tweet)


    return translated_tweets

# Function to clean tweets
def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        cleaned_tweet = extract_text(tweet)
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets


# Function to extract text
def extract_text(tweet):
    # Remove text before and including the @username
    tweet = re.sub(r'.*?@\w+:?', '', tweet)

    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet


# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


# Function to get sentiment scores for a list of tweets
def get_sentiments(tweets):
    sentiment_scores = [analyze_sentiment(tweet) for tweet in tweets]
    return sentiment_scores


# Function to analyze sentiment using Empath
def analyze_sentiment_empath(tweets, sentiment_scores, empath_scores):
    df = pd.DataFrame({'Tweet': tweets})
    df['sentiment_neg'] = [score['neg'] for score in sentiment_scores]
    df['sentiment_neu'] = [score['neu'] for score in sentiment_scores]
    df['sentiment_pos'] = [score['pos'] for score in sentiment_scores]
    df['sentiment_compound'] = [score['compound'] for score in sentiment_scores]
    df['vader_compound_rounded'] = 'neu'
    df.loc[df['sentiment_compound'] > 0, 'vader_compound_rounded'] = 'pos'
    df.loc[df['sentiment_compound'] < 0, 'vader_compound_rounded'] = 'neg'

    dfEmpath = pd.DataFrame.from_records(empath_scores)
    dfEmpath = dfEmpath.loc[:, (dfEmpath != 0).any(axis=0)]
    df = pd.concat([df, dfEmpath], axis=1)

    return df


# Function to get emotion totals

def get_emotion_totals(df, top_percent):
    # Filter out non-emotion columns
    emotion_columns = df.drop(columns=['Tweet', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound',
                                       'vader_compound_rounded'])

    # Calculate column-wise sum for emotion columns
    column_wise_sum = emotion_columns.sum(axis=0)

    # Create a dictionary mapping emotion names to their total scores
    emotion_totals = dict(column_wise_sum)

    # Calculate the number of items corresponding to the top percentage
    top_count = int(len(emotion_totals) * top_percent)

    # Sort the dictionary items by values in descending order
    sorted_items = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)

    # Get the top key-value pairs
    top_percent_items = sorted_items[:top_count]

    # Create a new dictionary for the top percentage key-value pairs
    top_percent_dict = dict(top_percent_items)

    return top_percent_dict


# Function to create a square matrix
def create_square_matrix(top_dict, df):
    # Create an empty square matrix represented as a DataFrame with named rows and columns
    column_names = list(top_dict.keys())
    square_matrix_df = pd.DataFrame(0, index=column_names, columns=column_names)

    # Iterate over the keys
    for key in column_names:
        # Count the occurrences of the key in the DataFrame
        count = df[key].sum()
        # Assign the count to the diagonal element in the square matrix
        square_matrix_df.loc[key, key] = count

    # Iterate over the combinations of keys
    for key1, key2 in combinations(column_names, 2):
        # Check if both keys are present in the dictionary
        if key1 in top_dict and key2 in top_dict:
            # Count the number of occurrences of the combination in the DataFrame
            count = ((df[key1] == 1) & (df[key2] == 1)).sum()
            # Assign the count to the corresponding entry in the square matrix
            square_matrix_df.loc[key1, key2] = count
            square_matrix_df.loc[key2, key1] = count  # Since it's an undirected graph, update both symmetric entries

    return square_matrix_df


# Function to plot graph
def plot_graph(square_matrix):
    # Convert square matrix to DataFrame
    df = pd.DataFrame(square_matrix)

    # Get emotion names as node labels
    node_labels = df.index.tolist()

    # Convert square matrix to numpy array
    matrix = np.array(square_matrix)

    # Create a graph from the numpy array
    G = nx.from_numpy_array(matrix)

    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))

    # Use Kamada-Kawai layout for positioning the nodes
    pos = nx.kamada_kawai_layout(G)

    # Draw the graph with circular nodes and no outlines
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels={i: label for i, label in enumerate(node_labels)}, node_color='skyblue', node_size=1000, font_size=10, font_weight='bold', edge_color='gray', width=1.5, node_shape='o', edgecolors='none')
    plt.title("Emotion Graph")
    plt.axis('off')  # Disable axis
    st.pyplot()
def read_from_file(txtfile):
    lines = txtfile.read().decode('utf-8').splitlines()
    print(lines)
    return lines



def main():
    st.title("Twitter Tweet Analyzer")
    # Your Streamlit UI code goes here
    toggle = st.checkbox("Manual Mode")
    if toggle:
        uploaded_file = st.file_uploader("Upload Text file", type=["txt"])
        top_percent_input = st.number_input("Enter the percentage for top items:", min_value=1, max_value=100, step=1,
                                            value=10)
        model_type = st.selectbox("Select Model", ['', 'Linear SVC', 'Naive Bayes Multinomial'])
        encoding_dict = {0: 'A', 1: 'AD', 2: 'I', 3: 'T'}

        bert_model_type = st.selectbox("Select Model", ['bert-base-uncased', 'bert-large-uncased'])
        if st.button("Analyze Tweets"):
            if uploaded_file:
                # Retrieve tweets
                tweets = read_from_file(uploaded_file)
                if tweets is not None:
                    st.success("Tweets retrieved successfully!")

                    # Translate tweets
                    translated_tweets = translate_tweets(tweets)

                    # Clean tweets
                    cleaned_tweets = clean_tweets(translated_tweets)

                    # Analyze sentiment
                    sentiment_scores = get_sentiments(cleaned_tweets)

                    # Initialize VADER sentiment analyzer
                    analyzer = SentimentIntensityAnalyzer()

                    # Analyze with Empath
                    lexicon = Empath()

                    empath_scores = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets]

                    # Convert cleaned tweets, sentiment scores, and empath scores to DataFrame
                    df = analyze_sentiment_empath(cleaned_tweets, sentiment_scores, empath_scores)

                    # Get emotion totals
                    top_percent = top_percent_input / 100

                    # Retrieve top emotions
                    top_emotions = get_emotion_totals(df, top_percent)

                    square_matrix = create_square_matrix(top_emotions, df)

                    # Display DataFrame
                    st.write("Analysis Results:")
                    st.write(df)
                    st.write(top_emotions)
                    st.write(square_matrix)
                    # Plot the graph
                    st.markdown("### Emotion Graph:")
                    plot_graph(square_matrix)

                    #model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])

                    if model_type == 'Linear SVC' or model_type == 'Naive Bayes Multinomial':
                        st.write(f"Model selected: {model_type}")
                        model_file = model_type + 'trained_model.pkl'
                        vectorizer_file = model_type + 'tfidf_vectorizer.pkl'

                        try:
                            model = joblib.load(model_file)
                            tfidf_vectorizer = joblib.load(vectorizer_file)

                            st.write("Model and TF-IDF vectorizer loaded successfully.")

                            # Convert the cleaned retrieved tweets to TF-IDF features
                            X_retrieved_tfidf = tfidf_vectorizer.transform(df['Tweet'])

                            # Predict intents using the loaded model
                            predictions = model.predict(X_retrieved_tfidf)

                            # Convert predictions to DataFrame
                            predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])

                            st.write(predictions_df)

                            label_encoder = joblib.load('label_encoder.pkl')

                            intent_encoded = predictions_df['intent_encoded']

                            predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)

                            mapping_values = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                            st.write(mapping_values)

                            # Concatenate 'tweet' column with predictions DataFrame
                            result_df = pd.concat([df, predictions_df], axis=1)

                            # Display result
                            st.write(result_df)

                        except Exception as e:
                            st.error(f"Error loading model: {e}")

                        intents = result_df['intent'].unique()

                        for intent in intents:
                            st.write(intent)

                            # Filter the dataframe for the current intent
                            df_intent = result_df[result_df['intent'] == intent]
                            st.write(df_intent)

                            cols_exclude = ['intent','intent_encoded']

                            modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]

                            # Retrieve top emotions
                            top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                            st.write(top_emotions_intent)

                            st.write('Square matrix for ', intent)
                            # Generate the square matrix for the current intent
                            intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                            st.write(intent_square_matrix)

                            # Plot the graph
                            st.write("### Emotion Graph for :", intent)
                            #
                            # # Plot the graph for the current intent

                            plot_graph(intent_square_matrix)
                    else:
                        st.write('\n')
                        st.write('No machine learning model selected.')

                    if bert_model_type == 'bert-base-uncased' or bert_model_type == 'bert-large-uncased':
                        st.write(f"Model selected: {bert_model_type}")
                        bert_model_file = bert_model_type + 'trained_bert_model.pth'
                        tokenizer = BertTokenizer.from_pretrained(bert_model_type)

                        # Check if CUDA is available
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # Get the current working directory
                        current_directory = os.getcwd()

                        try:
                            # Load the BERT model
                            saved_model_path = os.path.join(current_directory, bert_model_file)
                            bert_model = BertForSequenceClassification.from_pretrained(bert_model_type, num_labels=4)
                            bert_model.load_state_dict(torch.load(saved_model_path, map_location=device))
                            bert_model.eval()
                            st.write("BERT model loaded successfully.")
                            # Preprocess test data and create DataLoader
                            test_encodings = tokenizer(df['Tweet'].tolist(), truncation=True, padding=True,return_tensors='pt')
                            test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
                            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

                            # Evaluate the model
                            predictions = []

                            with torch.no_grad():
                                for batch in test_loader:
                                    input_ids, attention_mask = batch
                                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                                    logits = outputs.logits
                                    predictions.extend(logits.argmax(dim=1).cpu().numpy())

                            st.write('Got predictions')

                            # Convert predictions to DataFrame
                            predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])

                            st.write(predictions_df)

                            label_encoder = joblib.load('label_encoder.pkl')

                            intent_encoded = predictions_df['intent_encoded']

                            predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)

                            mapping_values = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                            st.write(mapping_values)

                            # Concatenate 'tweet' column with predictions DataFrame
                            result_df = pd.concat([df, predictions_df], axis=1)

                            # Display result
                            st.write(result_df)

                        except Exception as e:
                            st.error(f"Error loading BERT model: {e}")

                        intents = result_df['intent'].unique()

                        for intent in intents:
                            st.write(intent)

                            # Filter the dataframe for the current intent
                            df_intent = result_df[result_df['intent'] == intent]
                            st.write(df_intent)

                            cols_exclude = ['intent', 'intent_encoded']

                            modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]

                            # Retrieve top emotions
                            top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                            st.write(top_emotions_intent)

                            st.write('Square matrix for ', intent)
                            # Generate the square matrix for the current intent
                            intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                            st.write(intent_square_matrix)

                            # Plot the graph
                            st.write("### Emotion Graph for :", intent)
                            #
                            # # Plot the graph for the current intent

                            plot_graph(intent_square_matrix)
            else:
                st.error("Error occurred during tweet retrieval. Please check your input.")
        else:
            st.warning("Please upload a text file.")


    else:
        bearer_token = st.text_input("Enter your Twitter Bearer Token:", type="password")
        search_str = st.text_input("Enter keyword to search for tweets:")
        num_tweets = st.number_input("Enter number of tweets to retrieve:", min_value=1, max_value=500, step=1, value=10)
        top_percent_input = st.number_input("Enter the percentage for top items:", min_value=1, max_value=100, step=1, value=10)

        model_type = st.selectbox("Select Model", ['','Linear SVC', 'Naive Bayes Multinomial'])
        encoding_dict = {0: 'A', 1: 'AD', 2: 'I', 3: 'T'}

        bert_model_type = st.selectbox("Select Model", ['bert-base-uncased', 'bert-large-uncased'])

        if st.button("Analyze Tweets"):
            if bearer_token:
                # Retrieve tweets
                tweets = retrieve_tweets(bearer_token, search_str, num_tweets)
                if tweets is not None:
                    st.success("Tweets retrieved successfully!")

                    # Translate tweets
                    translated_tweets = translate_tweets(tweets)

                    # Clean tweets
                    cleaned_tweets = clean_tweets(translated_tweets)

                    # Analyze sentiment
                    sentiment_scores = get_sentiments(cleaned_tweets)

                    # Initialize VADER sentiment analyzer
                    analyzer = SentimentIntensityAnalyzer()

                    # Analyze with Empath
                    lexicon = Empath()

                    empath_scores = [lexicon.analyze(tweet, normalize=False) for tweet in cleaned_tweets]

                    # Convert cleaned tweets, sentiment scores, and empath scores to DataFrame
                    df = analyze_sentiment_empath(cleaned_tweets, sentiment_scores, empath_scores)

                    # Get emotion totals
                    top_percent = top_percent_input / 100

                    # Retrieve top emotions
                    top_emotions = get_emotion_totals(df, top_percent)

                    square_matrix = create_square_matrix(top_emotions, df)

                    # Display DataFrame
                    st.write("Analysis Results:")
                    st.write(df)
                    st.write(top_emotions)
                    st.write(square_matrix)
                    # Plot the graph
                    st.markdown("### Emotion Graph:")
                    plot_graph(square_matrix)

                    #model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])

                    if model_type == 'Linear SVC' or model_type == 'Naive Bayes Multinomial':
                        st.write(f"Model selected: {model_type}")
                        model_file = model_type + 'trained_model.pkl'
                        vectorizer_file = model_type + 'tfidf_vectorizer.pkl'

                        try:
                            model = joblib.load(model_file)
                            tfidf_vectorizer = joblib.load(vectorizer_file)

                            st.write("Model and TF-IDF vectorizer loaded successfully.")

                            # Convert the cleaned retrieved tweets to TF-IDF features
                            X_retrieved_tfidf = tfidf_vectorizer.transform(df['Tweet'])

                            # Predict intents using the loaded model
                            predictions = model.predict(X_retrieved_tfidf)

                            # Convert predictions to DataFrame
                            predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])

                            st.write(predictions_df)

                            label_encoder = joblib.load('label_encoder.pkl')

                            intent_encoded = predictions_df['intent_encoded']

                            predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)

                            mapping_values = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                            st.write(mapping_values)

                            # Concatenate 'tweet' column with predictions DataFrame
                            result_df = pd.concat([df, predictions_df], axis=1)

                            # Display result
                            st.write(result_df)

                        except Exception as e:
                            st.error(f"Error loading model: {e}")

                        intents = result_df['intent'].unique()

                        for intent in intents:
                            st.write(intent)

                            # Filter the dataframe for the current intent
                            df_intent = result_df[result_df['intent'] == intent]
                            st.write(df_intent)

                            cols_exclude = ['intent','intent_encoded']

                            modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]

                            # Retrieve top emotions
                            top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                            st.write(top_emotions_intent)

                            st.write('Square matrix for ', intent)
                            # Generate the square matrix for the current intent
                            intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                            st.write(intent_square_matrix)

                            # Plot the graph
                            st.write("### Emotion Graph for :", intent)
                            #
                            # # Plot the graph for the current intent

                            plot_graph(intent_square_matrix)
                    else:
                        st.write('\n')
                        st.write('No machine learning model selected.')

                    if bert_model_type == 'bert-base-uncased' or bert_model_type == 'bert-large-uncased':
                        st.write(f"Model selected: {bert_model_type}")
                        bert_model_file = bert_model_type + 'trained_bert_model.pth'
                        tokenizer = BertTokenizer.from_pretrained(bert_model_type)

                        # Check if CUDA is available
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        # Get the current working directory
                        current_directory = os.getcwd()

                        try:
                            # Load the BERT model
                            saved_model_path = os.path.join(current_directory, bert_model_file)
                            bert_model = BertForSequenceClassification.from_pretrained(bert_model_type, num_labels=4)
                            bert_model.load_state_dict(torch.load(saved_model_path, map_location=device))
                            bert_model.eval()
                            st.write("BERT model loaded successfully.")
                            # Preprocess test data and create DataLoader
                            test_encodings = tokenizer(df['Tweet'].tolist(), truncation=True, padding=True,return_tensors='pt')
                            test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
                            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

                            # Evaluate the model
                            predictions = []

                            with torch.no_grad():
                                for batch in test_loader:
                                    input_ids, attention_mask = batch
                                    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                                    outputs = bert_model(input_ids, attention_mask=attention_mask)
                                    logits = outputs.logits
                                    predictions.extend(logits.argmax(dim=1).cpu().numpy())

                            st.write('Got predictions')

                            # Convert predictions to DataFrame
                            predictions_df = pd.DataFrame(predictions, columns=['intent_encoded'])

                            st.write(predictions_df)

                            label_encoder = joblib.load('label_encoder.pkl')

                            intent_encoded = predictions_df['intent_encoded']

                            predictions_df['intent'] = label_encoder.inverse_transform(intent_encoded)

                            mapping_values = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

                            st.write(mapping_values)

                            # Concatenate 'tweet' column with predictions DataFrame
                            result_df = pd.concat([df, predictions_df], axis=1)

                            # Display result
                            st.write(result_df)

                        except Exception as e:
                            st.error(f"Error loading BERT model: {e}")

                        intents = result_df['intent'].unique()

                        for intent in intents:
                            st.write(intent)

                            # Filter the dataframe for the current intent
                            df_intent = result_df[result_df['intent'] == intent]
                            st.write(df_intent)

                            cols_exclude = ['intent', 'intent_encoded']

                            modified_df = df_intent.loc[:, ~df_intent.columns.isin(cols_exclude)]

                            # Retrieve top emotions
                            top_emotions_intent = get_emotion_totals(modified_df, top_percent)
                            st.write(top_emotions_intent)

                            st.write('Square matrix for ', intent)
                            # Generate the square matrix for the current intent
                            intent_square_matrix = create_square_matrix(top_emotions_intent, modified_df)
                            st.write(intent_square_matrix)

                            # Plot the graph
                            st.write("### Emotion Graph for :", intent)
                            #
                            # # Plot the graph for the current intent

                            plot_graph(intent_square_matrix)
            else:
                st.error("Error occurred during tweet retrieval. Please check your input.")
        else:
            st.warning("Please enter your Twitter Bearer Token.")

if __name__ == "__main__":
    main()