import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
import boto3
from io import StringIO
import datetime
import re

#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs


################# Twitter API Connection #######################
consumer_key = "xxx"
consumer_secret = "xxx"
access_token = "xxx-xxx"
access_token_secret = "xx"



# Use the above credentials to authenticate the API.

auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
auth.set_access_token( access_token , access_token_secret )
api = tweepy.API(auth)
################################################################

df = pd.DataFrame(columns=["Tweet_Date","Twitter_User","IsVerified","Tweet","Likes","RT",'User_location'])

# Write a Function to extract tweets:
def get_tweets(Topic,Count):
    i=0
    #my_bar = st.progress(100) # To track progress of Extracted tweets
    for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
        #time.sleep(0.1)
        #my_bar.progress(i)
        df.loc[i,"Tweet_Date"] = tweet.created_at
        df.loc[i,"Twitter_User"] = tweet.user.name
        df.loc[i,"IsVerified"] = tweet.user.verified
        df.loc[i,"Tweet"] = tweet.text
        df.loc[i,"Likes"] = tweet.favorite_count
        df.loc[i,"RT"] = tweet.retweet_count
        df.loc[i,"User_location"] = tweet.user.location
        #df.to_csv("TweetDataset.csv",index=False)
        #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
        i=i+1
        if i>Count:
            break
        else:
            pass

# Function to Clean the Tweet.
def clean_tweet(tweet):
    return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

    
# Funciton to analyze Sentiment
def analyze_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

lst = ['AMZN','MSFT','GOOG','CVS','UNH','HUM','COST','WMT','TGT','DELL','AAPL','HPQ','GM','TSLA']
dff = pd.DataFrame()

for i in lst:

    get_tweets(i , Count=200)

    df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))

    # Call function to get the Sentiments
    df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))

    df["Company"] = i

    dff = dff.append(df)

    dff = dff.fillna(0)

    # df.Tweet.fillna(value=0, inplace=True)

    # df.a.fillna(value=0, inplace=True)



dff = dff.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))


dff['Company'] = dff['Company'].str.replace(',', '')
dff['Company'] = dff['Company'].str.replace('\r', '')
dff['Company'] = dff['Company'].str.replace("\\", '')
dff['Company'].replace('', np.nan, inplace=True)

dff.dropna(subset=['Company'], inplace=True)

dff['Tweet'] = dff['Tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
dff['Tweet'] = dff['Tweet'].str.replace(',', '')
dff['Tweet'] = dff['Tweet'].str.replace('\r', '')
dff['Tweet'] = dff['Tweet'].str.replace("\\", '')
dff['Tweet'] = dff['Tweet'].str.strip().replace("", 'Unkown')
dff['Tweet'].replace('', np.nan, inplace=True)

dff.dropna(subset=['Tweet'], inplace=True)

dff['Twitter_User'] = dff['Twitter_User'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
dff['Twitter_User'] = dff['Twitter_User'].str.replace(',', '')
dff['Twitter_User'] = dff['Twitter_User'].str.replace('\r', '')
dff['Twitter_User'] = dff['Twitter_User'].str.replace("\\", '')
dff['Twitter_User'] = dff['Twitter_User'].str.strip().replace("", 'Unkown')
dff['Twitter_User'].replace('', np.nan, inplace=True)

dff.dropna(subset=['Twitter_User'], inplace=True)

dff['User_location'] = dff['User_location'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
dff['User_location'] = dff['User_location'].str.replace(',', '')
dff['User_location'] = dff['User_location'].str.replace('\r', '')
dff['User_location'] = dff['User_location'].str.replace("\\", '')
dff['User_location'] = dff['User_location'].str.strip().replace("", 'Unkown')
dff['User_location'].replace('', np.nan, inplace=True)


dff.dropna(subset=['User_location'], inplace=True)


print(dff)


# Upload to S3 Bucket

ACCESS_KEY_ID = 'xx'
ACCESS_SECRET_KEY = 'xx'

s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id= ACCESS_KEY_ID,
    aws_secret_access_key=ACCESS_SECRET_KEY
)
bucket = 'scrappednewsdata' # already created on S3

csv_buffer = StringIO()
dff.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'TwitterCSVFiles/TwitterData-'+str(datetime.datetime.now())+'.csv').put(Body=csv_buffer.getvalue())
#s3_resource.Object(bucket,'TwitterData.csv').put(Body=csv_buffer.getvalue())

print('Successful!!!!!')


