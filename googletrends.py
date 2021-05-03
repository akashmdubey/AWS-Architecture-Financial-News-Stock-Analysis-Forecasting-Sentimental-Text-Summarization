import pytrends
from pytrends.request import TrendReq
import pandas as pd
import time
import datetime
from datetime import datetime, date, time
import boto3
from io import StringIO

pytrend = TrendReq()
#pytrend.build_payload(kw_list=['AMZN','MSFT','GOOG','CVS','UNH','HUM','COST','WMT','TGT','DELL','AAPL','HPQ','GM','TSLA'], timeframe='today 5-y', geo = 'US')

# Cloud Service Providers
pytrend.build_payload(kw_list=['AMZN','MSFT','GOOG'], timeframe='today 5-y', geo = 'US')

interest_Cloud_Companies = pytrend.interest_over_time()
interest_Cloud_Companies['Date'] = interest_Cloud_Companies.index


# Healthcare Providers
pytrend = TrendReq()
pytrend.build_payload(kw_list=['CVS','UNH','HUM'], timeframe='today 5-y', geo = 'US')

interest_Healthcare_Companies = pytrend.interest_over_time()
interest_Healthcare_Companies['Date'] = interest_Healthcare_Companies.index
print(interest_Healthcare_Companies.head())

# Store Providers
pytrend = TrendReq()
pytrend.build_payload(kw_list=['COST','WMT','TGT'], timeframe='today 5-y', geo = 'US')

interest_Store_Companies = pytrend.interest_over_time()
interest_Store_Companies['Date'] = interest_Store_Companies.index
print(interest_Store_Companies.head())

# Computer Providers
pytrend = TrendReq()
pytrend.build_payload(kw_list=['DELL','AAPL','HPQ'], timeframe='today 5-y', geo = 'US')

interest_Computer_Companies = pytrend.interest_over_time()
interest_Computer_Companies['Date'] = interest_Computer_Companies.index

# Automobile Providers
pytrend = TrendReq()
pytrend.build_payload(kw_list=['GM','TSLA'], timeframe='today 5-y', geo = 'US')

interest_Automobile_Companies = pytrend.interest_over_time()
interest_Automobile_Companies['Date'] = interest_Automobile_Companies.index





# AWS Upload

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
#Cloud
interest_Cloud_Companies.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'GoogleTrendsData/GoogleTrends_Cloud_Companies.csv').put(Body=csv_buffer.getvalue())

#Healthcare
csv_buffer = StringIO()
interest_Healthcare_Companies.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'GoogleTrendsData/GoogleTrends_Healthcare_Companies.csv').put(Body=csv_buffer.getvalue())

#Store
csv_buffer = StringIO()
interest_Store_Companies.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'GoogleTrendsData/GoogleTrends_Store_Companies.csv').put(Body=csv_buffer.getvalue())

#Computer
csv_buffer = StringIO()
interest_Computer_Companies.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'GoogleTrendsData/GoogleTrends_Computer_Companies.csv').put(Body=csv_buffer.getvalue())

#Auto
csv_buffer = StringIO()
interest_Automobile_Companies.to_csv(csv_buffer, index = False, encoding='utf-8')
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket,'GoogleTrendsData/GoogleTrends_Auto_Companies.csv').put(Body=csv_buffer.getvalue())


print('Successful!!!!!')

