import streamlit as st
import altair as alt
from os import listdir
from os.path import isfile, join
from pydantic import BaseModel
import boto3
import json
import time
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import string
from datetime import datetime
from datetime import date
import requests as requests

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from pytrends.request import TrendReq

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
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#To Hide Warnings

from urllib.request import urlopen, Request
import bs4 
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from gensim.summarization import summarize

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

# data_dir = '/root/Assignment4/Assignment-Trial/Assignment-Trial/inference-data/'
# data_dir2 = '/root/Assignment4/Assignment-Trial/Assignment-Trial/fastAPIandStreamlit/awsdownload/'
data_dir3 = './awsdownloadstar/'

#companies = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
@st.cache
def load_data():
    #df = data.cars()

    return 0

def get_data(keyword):
    keyword = [keyword]
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keyword)
    df = pytrend.interest_over_time()
    df.drop(columns=['isPartial'], inplace=True)
    df.reset_index(inplace=True)
    df.columns = ["ds", "y"]
    return df

# make forecasts for a new period
def make_pred(df, periods):
    prophet_basic = Prophet()
    prophet_basic.fit(df)
    future = prophet_basic.make_future_dataframe(periods=periods)
    forecast = prophet_basic.predict(future)
    fig1 = prophet_basic.plot(forecast, xlabel="date", ylabel="trend", figsize=(10, 6))
    fig2 = prophet_basic.plot_components(forecast)
    forecast = forecast[["ds", "yhat"]]

    return forecast, fig1, fig2






def main():
    df = load_data()
    
    

    page = st.sidebar.selectbox('Choose a page',('Homepage', 'SignUp','Logout'))

    
    #page = st.sidebar.radio("Choose a page", ["Homepage", "SignUp"])
    if page == "Homepage":

        ACCESS_KEY_ID = 'xx'
        ACCESS_SECRET_KEY = 'xx'
        st.title('** Welcome to Team 3 CSYE !!!**')
        st.header('User Authentication')

        st.subheader('Please enter valid username password and Acess Token')
    
        usrName = st.text_input('Username')
        usrPassword = st.text_input('Password')
        acesstoken = st.text_input('Enter your Token')

        OTP = usrName + usrPassword
        dynamodb = boto3.resource('dynamodb',
                    aws_access_key_id=ACCESS_KEY_ID,
                    aws_secret_access_key=ACCESS_SECRET_KEY,
                    region_name='us-east-1')


        table = dynamodb.Table('users')

        response = table.scan()

        OTPD = response['Items']
        userlist = []
        toklist = []
        i = 0
        while i < len(OTPD):
            #print(OTP[i])
            x = OTPD[i]['login']
            y = OTPD[i]['acesstoken']
            #print(x)
            userlist.append(x)
            toklist.append(y)
            i=i+1

        

        
        if OTP in userlist and acesstoken in toklist :
            verified = "True"
            result = "Congratulations User Verified!!"
            page = st.sidebar.radio("Choose a page", ["Star Masked Data","Live News","Company Profile","Technical","Google Trends","Twitter Trends","Stock Future Prediction", "Meeting Summarization"])
            st.title(result)
            
            
            
            
            if page == "Star Masked Data":
                
                st.title("Star Data Using AWS Comprehend")
                user_input = st.text_input("Enter the name of the Company")
                user_input = user_input+".out"
                time.sleep(1.4)
            

                try:
                    with open(data_dir3 + user_input) as f:
                        st.text(f.read())
                except:
                    st.text("Company Does not Exist")


            elif page == "Google Trends":
                st.sidebar.write("""
                ## Choose a keyword and a prediction period 
                """)
                keyword = st.sidebar.text_input("Keyword", "Amazon")
                periods = st.sidebar.slider('Prediction time in days:', 7, 365, 90)
                details = st.sidebar.checkbox("Show details")

                # main section
                st.write("""
                # Welcome to Trend Predictor App
                ### This app predicts the **Google Trend** you want!
                """)
                st.image('https://s3.eu-west-2.amazonaws.com/cdn.howtomakemoneyfromhomeuk.com/wp-content/uploads/2020/10/Google-Trends.jpg',width=350, use_column_width=200)
                st.write("Evolution of interest:", keyword)

                df = get_data(keyword)
                forecast, fig1, fig2 = make_pred(df, periods)

                st.pyplot(fig1)
                    
                if details:
                    st.write("### Details :mag_right:")
                    st.pyplot(fig2)

            elif page == "Meeting Summarization":

                symbols = ['./Audio Files/Meeting 1.mp3','./Audio Files/Meeting 2.mp3', './Audio Files/Meeting 3.mp3', './Audio Files/Meeting 4.mp3']

                track = st.selectbox('Choose a the Meeting Audio',symbols)

                st.audio(track)
                data_dir = './inference-data/'

                ratiodata = st.text_input("Please Enter a Ratio you want summary by: ")
                if st.button("Generate a Summarized Version of the Meeting"):
                    time.sleep(2.4)
                    #st.success("This is the Summarized text of the Meeting Audio Files xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  xxxxxxgeeeeeeeeeeeeeee eeeeeeeeeeeeeehjjjjjjjjjjjjjjjsdbjhvsdk vjbsdkvjbsdvkb skbdv")
                    
                    
                    if track == "./Audio Files/Meeting 2.mp3":
                        user_input = "NKE"
                        time.sleep(1.4)
                        try:
                            with open(data_dir + user_input) as f:
                                st.success(summarize(f.read(), ratio=float(ratiodata)))          
                                #print()
                                st.warning("Sentiment: Negative")
                        except:
                            st.text("Company Does not Exist")

                    else:
                        user_input = "AGEN"
                        time.sleep(1.4)
                        try:
                            with open(data_dir + user_input) as f:
                                st.success(summarize(f.read(), ratio=float(ratiodata)))          
                                #print()
                                st.success("Sentiment: Positive")
                        except:
                            st.text("Company Does not Exist")

            elif page == "Twitter Trends":


                st.write("""
                # Welcome to Twitter Sentiment App
                ### This app predicts the **Twitter Sentiments** you want!
                """)
                st.image('https://assets.teenvogue.com/photos/56b4f21327a088e24b967bb6/3:2/w_531,h_354,c_limit/twitter-gifs.gif',width=250, use_column_width=200)

                
                #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

                ################# Twitter API Connection #######################
                consumer_key = "xx"
                consumer_secret = "xx"
                access_token = "xx"
                access_token_secret = "xx"



                # Use the above credentials to authenticate the API.

                auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
                auth.set_access_token( access_token , access_token_secret )
                api = tweepy.API(auth)
                ################################################################
    
                df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])

                # Write a Function to extract tweets:
                def get_tweets(Topic,Count):
                    i=0
                    #my_bar = st.progress(100) # To track progress of Extracted tweets
                    for tweet in tweepy.Cursor(api.search, q=Topic,count=100, lang="en",exclude='retweets').items():
                        #time.sleep(0.1)
                        #my_bar.progress(i)
                        df.loc[i,"Date"] = tweet.created_at
                        df.loc[i,"User"] = tweet.user.name
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

                #Function to Pre-process data for Worlcloud
                def prepCloud(Topic_text,Topic):
                    Topic = str(Topic).lower()
                    Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
                    Topic = re.split("\s+",str(Topic))
                    stopwords = set(STOPWORDS)
                    stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
                    ###
                    text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
                    return text_new
                
                # Collect Input from user :
                Topic = str()
                Topic = str(st.sidebar.text_input("Enter the topic you are interested in (Press Enter once done)"))     
                
                if len(Topic) > 0 :
                    
                    # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
                    with st.spinner("Please wait, Tweets are being extracted"):
                        get_tweets(Topic , Count=200)
                    st.success('Tweets have been Extracted !!!!')    
                    
                
                    # Call function to get Clean tweets
                    df['clean_tweet'] = df['Tweet'].apply(lambda x : clean_tweet(x))
                
                    # Call function to get the Sentiments
                    df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
                    
                    
                    # Write Summary of the Tweets
                    st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
                    st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
                    st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
                    st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))
                    
                    # See the Extracted Data : 
                    if st.button("See the Extracted Data"):
                        #st.markdown(html_temp, unsafe_allow_html=True)
                        st.success("Below is the Extracted Data :")
                        st.write(df.head(50))
                    
                    
                    # get the countPlot
                    if st.button("Get Count Plot for Different Sentiments"):
                        st.success("Generating A Count Plot")
                        st.subheader(" Count Plot for Different Sentiments")
                        st.write(sns.countplot(df["Sentiment"], palette="Blues"))
                        st.pyplot()
                    
                    # Piechart 
                    if st.button("Get Pie Chart for Different Sentiments"):
                        st.success("Generating A Pie Chart")
                        a=len(df[df["Sentiment"]=="Positive"])
                        b=len(df[df["Sentiment"]=="Negative"])
                        c=len(df[df["Sentiment"]=="Neutral"])
                        d=np.array([a,b,c])
                        explode = (0.1, 0.0, 0.1)
                        st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative","Neutral"],autopct='%1.2f%%'))
                        st.pyplot()
                        
                        
                    # get the countPlot Based on Verified and unverified Users
                    if st.button("Get Count Plot Based on Verified and unverified Users"):
                        st.success("Generating A Count Plot (Verified and unverified Users)")
                        st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
                        st.write(sns.countplot(df["Sentiment"],hue=df.IsVerified))
                        st.pyplot()
                    
                    
                    ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud
                    
                    
                    # Create a Worlcloud
                    if st.button("Get WordCloud for all things said about {}".format(Topic)):
                        st.success("Generating A WordCloud for all things said about {}".format(Topic))
                        text = " ".join(review for review in df.clean_tweet)
                        stopwords = set(STOPWORDS)
                        text_newALL = prepCloud(text,Topic)
                        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=75, colormap="Blues", background_color="black").generate(text_newALL)
                        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                        st.pyplot()
                    
                    
                    #Wordcloud for Positive tweets only
                    if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
                        st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                        text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
                        stopwords = set(STOPWORDS)
                        text_new_positive = prepCloud(text_positive,Topic)
                        #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
                        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=75, colormap="Greens", background_color="black").generate(text_new_positive)
                        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                        st.pyplot()
                    
                    
                    #Wordcloud for Negative tweets only       
                    if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
                        st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                        text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
                        stopwords = set(STOPWORDS)
                        text_new_negative = prepCloud(text_negative,Topic)
                        #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
                        wordcloud = WordCloud(stopwords=stopwords,max_words=800,max_font_size=75, colormap="Reds", background_color="black").generate(text_new_negative)
                        st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                        st.pyplot()
                    
                    
                    
                    
                    
            
                #st.sidebar.subheader("Scatter-plot setup")
                #box1 = st.sidebar.selectbox(label= "X axis", options = numeric_columns)
                #box2 = st.sidebar.selectbox(label="Y axis", options=numeric_columns)
                #sns.jointplot(x=box1, y= box2, data=df, kind = "reg", color= "red")
                #st.pyplot()



                
            elif page == "Stock Future Prediction":
                snp500 = pd.read_csv("./Datasets/SP500.csv")
                symbols = snp500['Symbol'].sort_values().tolist()   

                ticker = st.sidebar.selectbox(
                    'Choose a S&P 500 Stock',
                symbols)

                START = "2015-01-01"
                TODAY = date.today().strftime("%Y-%m-%d")

                st.title('Stock Forecast App')

                st.image('https://media2.giphy.com/media/JtBZm3Getg3dqxK0zP/giphy-downsized-large.gif',width=250, use_column_width=200)

                # stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'W', 'TSLA')
                # selected_stock = st.selectbox('Select dataset for prediction', stocks)

                n_years = st.slider('Years of prediction:', 1, 4)
                period = n_years * 365

                st.title('Stock Forecast App To Do part in stockapp.py')

                data_load_state = st.text('Loading data...')

                data = yf.download(ticker, START, TODAY)
                data.reset_index(inplace=True)
                data_load_state.text('Loading data... done!')

                st.subheader('Raw data')
                st.write(data.tail())

                # Plot raw data
                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
                    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)
                    
                plot_raw_data()

                # Predict forecast with Prophet.
                df_train = data[['Date','Close']]
                df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

                m = Prophet()
                m.fit(df_train)
                future = m.make_future_dataframe(periods=period)
                forecast = m.predict(future)

                # Show and plot forecast
                st.subheader('Forecast data')
                st.write(forecast.tail())
                    
                st.write(f'Forecast plot for {n_years} years')
                fig1 = plot_plotly(m, forecast)
                st.plotly_chart(fig1)

                st.write("Forecast components")
                fig2 = m.plot_components(forecast)
                st.write(fig2)
            
            elif page == "Technical":
                snp500 = pd.read_csv("./Datasets/SP500.csv")
                symbols = snp500['Symbol'].sort_values().tolist()   

                ticker = st.sidebar.selectbox(
                    'Choose a S&P 500 Stock',
                symbols)

                stock = yf.Ticker(ticker)

                def calcMovingAverage(data, size):
                    df = data.copy()
                    df['sma'] = df['Adj Close'].rolling(size).mean()
                    df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
                    df.dropna(inplace=True)
                    return df
            
                def calc_macd(data):
                    df = data.copy()
                    df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
                    df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
                    df['macd'] = df['ema12'] - df['ema26']
                    df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
                    df.dropna(inplace=True)
                    return df

                def calcBollinger(data, size):
                    df = data.copy()
                    df["sma"] = df['Adj Close'].rolling(size).mean()
                    df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
                    df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
                    df["width"] = df["bolu"] - df["bold"]
                    df.dropna(inplace=True)
                    return df

                st.title('Technical Indicators')
                st.subheader('Moving Average')
                
                coMA1, coMA2 = st.beta_columns(2)
                
                with coMA1:
                    numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
                
                with coMA2:
                    windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  
                    

                start = dt.datetime.today()-dt.timedelta(numYearMA * 365)
                end = dt.datetime.today()
                dataMA = yf.download(ticker,start,end)
                df_ma = calcMovingAverage(dataMA, windowSizeMA)
                df_ma = df_ma.reset_index()
                    
                figMA = go.Figure()
                
                figMA.add_trace(
                        go.Scatter(
                                x = df_ma['Date'],
                                y = df_ma['Adj Close'],
                                name = "Prices Over Last " + str(numYearMA) + " Year(s)"
                            )
                    )
                
                figMA.add_trace(
                            go.Scatter(
                                    x = df_ma['Date'],
                                    y = df_ma['sma'],
                                    name = "SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                                )
                        )
                
                figMA.add_trace(
                            go.Scatter(
                                    x = df_ma['Date'],
                                    y = df_ma['ema'],
                                    name = "EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
                                )
                        )
                
                figMA.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))
                
                figMA.update_layout(legend_title_text='Trend')
                figMA.update_yaxes(tickprefix="$")
                
                st.plotly_chart(figMA, use_container_width=True)  
                
                st.subheader('Moving Average Convergence Divergence (MACD)')
                numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 
                
                startMACD = dt.datetime.today()-dt.timedelta(numYearMACD * 365)
                endMACD = dt.datetime.today()
                dataMACD = yf.download(ticker,startMACD,endMACD)
                df_macd = calc_macd(dataMACD)
                df_macd = df_macd.reset_index()
                
                figMACD = make_subplots(rows=2, cols=1,
                                    shared_xaxes=True,
                                    vertical_spacing=0.01)
                
                figMACD.add_trace(
                        go.Scatter(
                                x = df_macd['Date'],
                                y = df_macd['Adj Close'],
                                name = "Prices Over Last " + str(numYearMACD) + " Year(s)"
                            ),
                        row=1, col=1
                    )
                
                figMACD.add_trace(
                        go.Scatter(
                                x = df_macd['Date'],
                                y = df_macd['ema12'],
                                name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"
                            ),
                        row=1, col=1
                    )
                
                figMACD.add_trace(
                        go.Scatter(
                                x = df_macd['Date'],
                                y = df_macd['ema26'],
                                name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"
                            ),
                        row=1, col=1
                    )
                
                figMACD.add_trace(
                        go.Scatter(
                                x = df_macd['Date'],
                                y = df_macd['macd'],
                                name = "MACD Line"
                            ),
                        row=2, col=1
                    )
                
                figMACD.add_trace(
                        go.Scatter(
                                x = df_macd['Date'],
                                y = df_macd['signal'],
                                name = "Signal Line"
                            ),
                        row=2, col=1
                    )
                
                figMACD.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
                
                figMACD.update_yaxes(tickprefix="$")
                st.plotly_chart(figMACD, use_container_width=True)
                
                st.subheader('Bollinger Band')
                coBoll1, coBoll2 = st.beta_columns(2)
                with coBoll1:
                    numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
                    
                with coBoll2:
                    windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
                
                startBoll= dt.datetime.today()-dt.timedelta(numYearBoll * 365)
                endBoll = dt.datetime.today()
                dataBoll = yf.download(ticker,startBoll,endBoll)
                df_boll = calcBollinger(dataBoll, windowSizeBoll)
                df_boll = df_boll.reset_index()
                figBoll = go.Figure()
                figBoll.add_trace(
                        go.Scatter(
                                x = df_boll['Date'],
                                y = df_boll['bolu'],
                                name = "Upper Band"
                            )
                    )
                
                
                figBoll.add_trace(
                            go.Scatter(
                                    x = df_boll['Date'],
                                    y = df_boll['sma'],
                                    name = "SMA" + str(windowSizeBoll) + " Over Last " + str(numYearBoll) + " Year(s)"
                                )
                        )
                
                
                figBoll.add_trace(
                            go.Scatter(
                                    x = df_boll['Date'],
                                    y = df_boll['bold'],
                                    name = "Lower Band"
                                )
                        )
                
                figBoll.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="left",
                    x=0
                ))
                
                figBoll.update_yaxes(tickprefix="$")
                st.plotly_chart(figBoll, use_container_width=True)

            elif page == "Live News":

                st.image('https://www.visitashland.com/files/latestnews.jpg',width=250, use_column_width=200)

                snp500 = pd.read_csv("./Datasets/SP500.csv")
                symbols = snp500['Symbol'].sort_values().tolist()   

                ticker = st.sidebar.selectbox(
                    'Choose a S&P 500 Stock',
                symbols)

                if st.button("See Latest News about "+ticker+""):

                    st.header('Latest News') 

                    def newsfromfizviz(temp):

                        # time.sleep(5)

                        finwiz_url = 'https://finviz.com/quote.ashx?t='


                        news_tables = {}
                        tickers = [temp]

                        for ticker in tickers:
                            url = finwiz_url + ticker
                            req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
                            response = urlopen(req)    
                            # Read the contents of the file into 'html'
                            html = BeautifulSoup(response)
                            # Find 'news-table' in the Soup and load it into 'news_table'
                            news_table = html.find(id='news-table')
                            # Add the table to our dictionary
                            news_tables[ticker] = news_table

                        parsed_news = []

                        # Iterate through the news
                        for file_name, news_table in news_tables.items():
                            # Iterate through all tr tags in 'news_table'
                            for x in news_table.findAll('tr'):
                                # read the text from each tr tag into text
                                # get text from a only
                                text = x.a.get_text() 
                                # splite text in the td tag into a list 
                                date_scrape = x.td.text.split()
                                # if the length of 'date_scrape' is 1, load 'time' as the only element

                                if len(date_scrape) == 1:
                                    time = date_scrape[0]

                                # else load 'date' as the 1st element and 'time' as the second    
                                else:
                                    date = date_scrape[0]
                                    time = date_scrape[1]
                                # Extract the ticker from the file name, get the string up to the 1st '_'  
                                ticker = file_name.split('_')[0]

                                # Append ticker, date, time and headline as a list to the 'parsed_news' list
                                parsed_news.append([ticker, date, time, text])

                        


                        # Instantiate the sentiment intensity analyzer
                        vader = SentimentIntensityAnalyzer()

                        # Set column names
                        columns = ['ticker', 'date', 'time', 'headline']

                        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
                        parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

                        # Iterate through the headlines and get the polarity scores using vader
                        scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

                        # Convert the 'scores' list of dicts into a DataFrame
                        scores_df = pd.DataFrame(scores)

                        # Join the DataFrames of the news and the list of dicts
                        parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

                        # Convert the date column from string to datetime
                        parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
                        
                        parsed_and_scored_news['Sentiment'] = np.where(parsed_and_scored_news['compound'] > 0, 'Positive', (np.where(parsed_and_scored_news['compound'] == 0, 'Neutral', 'Negative')))

                        return parsed_and_scored_news

                    df = newsfromfizviz(ticker)
                    df_pie = df[['Sentiment','headline']].groupby('Sentiment').count()
                    fig = px.pie(df_pie, values=df_pie['headline'], names=df_pie.index, color=df_pie.index, color_discrete_map={'Positive':'green', 'Neutral':'darkblue', 'Negative':'red'})

                    st.subheader('Dataframe with Latest News')
                    st.dataframe(df)

                    st.subheader('Latest News Sentiment Distribution using Pie Chart') 
                    st.plotly_chart(fig)

                    plt.rcParams['figure.figsize'] = [11, 5]

                    # Group by date and ticker columns from scored_news and calculate the mean
                    mean_scores = df.groupby(['ticker','date']).mean()

                    # Unstack the column ticker
                    mean_scores = mean_scores.unstack()

                    # Get the cross-section of compound in the 'columns' axis
                    mean_scores = mean_scores.xs('compound', axis="columns").transpose()

                    # Plot a bar chart with pandas
                    mean_scores.plot(kind = 'bar')

                    plt.grid()

                    st.set_option('deprecation.showPyplotGlobalUse', False)

                    st.subheader('Sentiments over Time')
                    st.pyplot()
            
            elif page == "Company Profile":
                snp500 = pd.read_csv("./Datasets/SP500.csv")
                symbols = snp500['Symbol'].sort_values().tolist()   

                ticker = st.sidebar.selectbox(
                    'Choose a S&P 500 Stock',
                symbols)

                stock = yf.Ticker(ticker)
                stock = yf.Ticker(ticker)
                info = stock.info 
                st.title('Company Profile')
                st.subheader(info['longName']) 
                st.markdown('** Sector **: ' + info['sector'])
                st.markdown('** Industry **: ' + info['industry'])
                st.markdown('** Phone **: ' + info['phone'])
                st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
                st.markdown('** Website **: ' + info['website'])
                st.markdown('** Business Summary **')
                st.info(info['longBusinessSummary'])
                    
                fundInfo = {
                        'Enterprise Value (USD)': info['enterpriseValue'],
                        'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
                        'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
                        'Net Income (USD)': info['netIncomeToCommon'],
                        'Profit Margin Ratio': info['profitMargins'],
                        'Forward PE Ratio': info['forwardPE'],
                        'PEG Ratio': info['pegRatio'],
                        'Price to Book Ratio': info['priceToBook'],
                        'Forward EPS (USD)': info['forwardEps'],
                        'Beta ': info['beta'],
                        'Book Value (USD)': info['bookValue'],
                        'Dividend Rate (%)': info['dividendRate'], 
                        'Dividend Yield (%)': info['dividendYield'],
                        'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
                        'Payout Ratio': info['payoutRatio']
                    }
                
                fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
                fundDF = fundDF.rename(columns={0: 'Value'})
                st.subheader('Fundamental Info') 
                st.table(fundDF)
                
                st.subheader('General Stock Info') 
                st.markdown('** Market **: ' + info['market'])
                st.markdown('** Exchange **: ' + info['exchange'])
                st.markdown('** Quote Type **: ' + info['quoteType'])
                
                start = dt.datetime.today()-dt.timedelta(2 * 365)
                end = dt.datetime.today()
                df = yf.download(ticker,start,end)
                df = df.reset_index()
                fig = go.Figure(
                        data=go.Scatter(x=df['Date'], y=df['Adj Close'])
                    )
                fig.update_layout(
                    title={
                        'text': "Stock Prices Over Past Two Years",
                        'y':0.9,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})
                st.plotly_chart(fig, use_container_width=True)
                
                marketInfo = {
                        "Volume": info['volume'],
                        "Average Volume": info['averageVolume'],
                        "Market Cap": info["marketCap"],
                        "Float Shares": info['floatShares'],
                        "Regular Market Price (USD)": info['regularMarketPrice'],
                        'Bid Size': info['bidSize'],
                        'Ask Size': info['askSize'],
                        "Share Short": info['sharesShort'],
                        'Short Ratio': info['shortRatio'],
                        'Share Outstanding': info['sharesOutstanding']
                
                    }
                
                marketDF = pd.DataFrame(data=marketInfo, index=[0])
                st.table(marketDF)


        else:
            verified = "False"
            result = "Please enter valid Username, Password and Acess Token!!"
    
            st.title(result)

    



    
    elif page == "Logout":
        st.header("You are Logged Out")
        st.subheader("Please Go to Homepage to Login")
        st.balloons()

    elif page == "SignUp":
            signusrName = st.text_input('Username')
            signusrPassword = st.text_input('Password')
            signusrEmail = st.text_input('Email')
            #signusrPasswordRepeat = st.text_input('Repeat Password')
            #accesstoken = st.text_input('Enter your Access Token')


            userOPT = signusrName + signusrPassword      
            
            ACCESS_KEY_ID = 'xx'
            ACCESS_SECRET_KEY = 'xx'

# dynamodb = boto3.resource('dynamodb',region_name='REGION', ) 
            dynamodb = boto3.resource('dynamodb',
                    aws_access_key_id=ACCESS_KEY_ID,
                    aws_secret_access_key=ACCESS_SECRET_KEY,
                    region_name='us-east-1')


            dynamoTable = dynamodb.Table('users')

            # Generating a Random Token for API Key
            letters = string.ascii_lowercase
            tok = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(40))

            print('######################## YOUR UNIQUE TOKEN ######################')
            print(tok)
        

            # Code to Put the Item
            if signusrName == "" or signusrPassword == "" or signusrEmail == "":
                st.text('Please enter Valid Values')

            else:
                st.title('Congratulations, Your Account is created Successfully!')
                st.title('Your Unique Token from AWS Cognito is Sent Via Email: ')
                #st.title(tok)
                import smtplib

                conn = smtplib.SMTP('smtp.gmail.com',587)

                conn.ehlo()

                conn.starttls()

                conn.login('jayshilcsyebigdata@gmail.com','Admin1234!')

                conn.sendmail('jayshilcsyebigdata@gmail.com',signusrEmail,'Subject: This is Your Access Token for FAST Stock App \n\n Dear '+signusrName+', \n Welcome to FAST Stock App \n Your Access Token is :'+tok+'')

                conn.quit()
                

                dynamoTable.put_item(
                    Item={
                        'login': userOPT,
                        'acesstoken': tok
                    }
                )


            

if __name__ == "__main__":
    main()
