import requests
import re
import xlsxwriter
import streamlit as st
from datetime import datetime,timedelta
# Set up your Twitter API bearer token
keyword = 'news -is:retweet'
filename= 'tweet.xlsx'
workbook = xlsxwriter.Workbook(filename)
worksheet = workbook.add_worksheet()
def retrieve_tweets(bearer_token, search_str, num_tweets, start_time, end_time):
    max_results_per_request = 15
    tweets = []

    # Define the Twitter API endpoint for searching tweets with pagination
    # define twitter api request to get tweets for a specific keyword within the date range provided and response should include created time.
    endpoint = f"https://api.twitter.com/2/tweets/search/recent?query={search_str}&max_results={max_results_per_request}&tweet.fields=created_at&expansions=author_id&start_time={start_time}&end_time={end_time}"
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
        print(data)
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
    #print(tweets)
    cleaned_tweets = []
    #write to excel sheet
    worksheet.write('A1', 'Tweet ID')
    worksheet.write('B1', 'Tweet')
    worksheet.write('C1', 'Time Created')
    worksheet.write('D1', 'Author ID')

    #iteration variable for excel sheet
    itr=2
    # Iterate over the tweets and print the tweet text
    for tweet in tweets:
        t = tweet['text']
        # Remove URLs, mentions, and hashtags from the tweet text
        t = re.sub(r"(?:\@|https?\://|\#)\S+", "", t)
        time_created= tweet['created_at']
        tweet_id= tweet['id']
        author= tweet['author_id']

        #print(t)
        #write to excel sheet
        worksheet.write('A' + str(itr), tweet_id)
        worksheet.write('B'+str(itr),t)
        worksheet.write('C'+str(itr),time_created)
        worksheet.write('D'+str(itr),author)
        cleaned_tweets.append(t)
        itr+=1
    workbook.close()
    print(len(cleaned_tweets))
    return cleaned_tweets
if __name__ == "__main__":
    st.title("Twitter Tweet Downloader ")
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAKFlsAEAAAAAachWrmQFbqeL3csdRLCkUhvd5kI%3D5K0WVdur2Z2bAUHKkZEzOkZOqD8inDTA4jUFC2ZnRXfruH9o8b"
    keyword= st.text_input("Enter the keyword")
    keyword= keyword+" -is:retweet"
    total_tweets= st.number_input("Enter the number of tweets to download",min_value=1, max_value=100)
    today = datetime.utcnow()
    min_date = today - timedelta(days=7)
    max_date = today
    start_date= st.date_input("Enter the start date",min_value=min_date,max_value=max_date)
    start_time= st.time_input("Enter the start time")
    end_date= st.date_input("Enter the end date",min_value=min_date,max_value=max_date)
    end_time= st.time_input("Enter the end time")
    start_date= str(start_date)+"T"+str(start_time)+"Z"
    end_date= str(end_date)+"T"+str(end_time)+"Z"
    if st.button("Download"):
        retrieve_tweets(bearer_token, keyword, total_tweets, start_date, end_date)
        st.success("Downloaded Successfully")





