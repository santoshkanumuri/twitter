import streamlit as st
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import nltk
import ssl
import numpy as np
import tweepy
from googletrans import Translator  # Assuming you're using the googletrans library
import networkx as nx
import matplotlib.pyplot as plt
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from itertools import combinations
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
import joblib


# Download stopwords if not already downloaded
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_data(df, column_name, threshold_percent):
    # Calculate value counts for the specified column
    value_counts = df[column_name].value_counts(normalize=True)

    # Get the values to drop based on the threshold percentage
    values_to_drop = value_counts[value_counts <= threshold_percent / 100].index

    # Filter the DataFrame
    df_filtered = df[~df[column_name].isin(values_to_drop)]

    return df_filtered

def label_encode_df(df):

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Extract the 'intent' column for label encoding
    intents = df['intent']

    # Fit and transform the 'intent' column
    df['intent_encoded'] = label_encoder.fit_transform(intents)

    # Save the LabelEncoder to a file
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # Get the mapping values (original labels and their corresponding encoded values)
    mapping_values = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return df, label_encoder, mapping_values


def resample_data(df, target_column, random_state=42):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    # Splitting the data into features (X) and target variable (y)
    X = df.drop(columns=[target_column])  # Assuming 'intent' is the target variable
    y = df[target_column]

    # Resampling the dataset to address class imbalance
    ros = RandomOverSampler(random_state=random_state)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled

def preprocess_and_lemmatize_text(text):
    # Set of punctuation characters
    punctuation = set(string.punctuation)

    # Set of stopwords
    stop_words = set(stopwords.words('english'))

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define regex pattern to match extra symbols
    extra_symbols_pattern = r'[^\w\s]'

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords, punctuation, and extra symbols
    filtered_tokens = [re.sub(extra_symbols_pattern, '', token.lower()) for token in tokens if
                       token.lower() not in punctuation and token.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the lemmatized tokens back into a string
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text

def train_ML_models(resample_df, model_type=None, test_size=0.3, random_state=42):

    if model_type is None:
        raise ValueError("Model type not specified.")

    X = resample_df['text']
    y = resample_df['intent_encoded']
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert text data to TF-IDF features
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Initialize the model
    if model_type == 'Linear SVC':
        model = LinearSVC()
    elif model_type == 'Naive Bayes Multinomial':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model type. Please choose 'Linear SVC' or 'Naive Bayes Multinomial'.")

    # Train the model
    model.fit(X_train_tfidf, y_train)
    joblib.dump(model, model_type+'trained_model.pkl')
    joblib.dump(tfidf_vectorizer, model_type+'tfidf_vectorizer.pkl')

    # # Predict on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, model

def train_bert_model(train_df, model_name):


    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # Tokenize the text data and create DataLoader
    def tokenize_text(text, max_length=109):
        return tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    # Split the dataset
    train_encodings = tokenize_text(list(train_df['text']))

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_df['intent_encoded'].values,dtype =torch.long))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Training
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    st.write("Training started...")
    for epoch in range(3):
        model.train()
        st.write('Epoch ', epoch)

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    st.write("Training completed!")
    return model

def evaluate_saved_bert_model(test_df,model_name):

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Get the current working directory
    current_directory = os.getcwd()


    # Specify the filename of the saved model
    filename = model_name + "trained_bert_model.pth"

    # Construct the full path to the saved model
    saved_model_path = os.path.join(current_directory, filename)

    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=4)
    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    st.write('model evaluation done')

    # # Preprocess test data and create DataLoader
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, return_tensors='pt')
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_df['intent_encoded'].values,dtype = torch.long))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Evaluate the model
    prediction = []
    true_label = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    st.write('Model is testing df ')
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction.extend(logits.argmax(dim=1).cpu().numpy())
            true_label.extend(labels.cpu().numpy())

    st.write('Got predictions')

    accuracy = accuracy_score(true_label, prediction)
    #classification_rep = classification_report(true_label, prediction)

    return accuracy,prediction


def main():
    st.title("Twitter Scraper")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display first few rows of the dataframe
        st.write("First few rows of the uploaded data:")

        threshold_percent = 5
        # Preprocess the data
        df_processed = preprocess_data(df, 'intent', threshold_percent)

        st.write(df_processed.head())
        st.write(df_processed.shape)

        #Encode the Data

        df_encoded,encoder,mapping_values = label_encode_df(df_processed)

        st.write(df_encoded.head())
        st.write(df_encoded.shape)
        st.write(encoder)
        st.write(mapping_values)

         # Resample the data
        X_resampled, y_resampled = resample_data(df_encoded, target_column='intent_encoded')

        # Display shape of resampled data
        st.write("Shape of Resampled Features (X):", X_resampled.shape)
        #
        # Display value counts of resampled target variable
        st.write("Value Counts of Resampled Target Variable (y):")
        st.write(pd.Series(y_resampled).value_counts())
        #
        # Apply text preprocessing to the 'text' column of X_resampled
        X_resampled['text'] = X_resampled['text'].apply(preprocess_and_lemmatize_text)

        # Display first few rows of resampled data
        st.write("First few rows of Resampled Data:")
        st.write(X_resampled.head())

        resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

        st.write(resampled_df.shape)

        # Split the dataset into training and testing sets
        train_df, test_df = train_test_split(resampled_df, test_size=0.25, random_state=42)

        st.write('Machine Learning Models')
        model_type = st.selectbox("Select Model", ['Linear SVC', 'Naive Bayes Multinomial'])

        if model_type == 'Linear SVC' or model_type == 'Naive Bayes Multinomial':
            st.write('Model' + model_type + ' is running')
            accuracy, ml_model = train_ML_models(resampled_df, model_type=model_type)
            st.write(f"Accuracy ({model_type}):", accuracy*100)


        st.write('BERT Models')
        select_type = st.selectbox("Select Model", ['Train BERT models', 'Load trained BERT models'])

        bert_model_type = st.selectbox("Select Model", ['bert-base-uncased', 'bert-large-uncased'])


        if select_type == 'Train BERT models':
            st.write('Calling BERT Function')
            trained_model = train_bert_model(train_df, model_name= bert_model_type)
            st.write("BERT model trained and saved successfully!")

            # Save the trained BERT model
            torch.save(trained_model.state_dict(), bert_model_type + "trained_bert_model.pth")
            st.write('Bert Model Saved succesfully')

        elif select_type =='Load trained BERT models':
            accuracy_test,predictions = evaluate_saved_bert_model(test_df, model_name= bert_model_type)
            st.write('Model loaded succesfully and  Training accuracy is ', accuracy_test*100)



if __name__ == "__main__":
    main()
