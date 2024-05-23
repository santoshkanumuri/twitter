import requests
import re
import time
from googletrans import Translator
# Set up your Twitter API bearer token
bearer_token = "AAAAAAAAAAAAAAAAAAAAAKFlsAEAAAAAachWrmQFbqeL3csdRLCkUhvd5kI%3D5K0WVdur2Z2bAUHKkZEzOkZOqD8inDTA4jUFC2ZnRXfruH9o8b"

keyword = 'news -is:retweet'
total_tweets = 101
def retrieve_tweets(bearer_token, search_str=keyword, num_tweets=total_tweets):
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





