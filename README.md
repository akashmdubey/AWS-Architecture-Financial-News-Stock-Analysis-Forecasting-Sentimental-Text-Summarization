## Project
### FAST - Financial Analytics with Stock Prediction and Timeseries Forecasting

## Project Report:
https://codelabs-preview.appspot.com/?file_id=1qxniFjwkDir6NT17KkvS1zDbmIgawcrEEwbbfCtAk8k#1


## Project Proposal:
https://codelabs-preview.appspot.com/?file_id=11_uaC--B3Yz0_ux-d8Pb0SZlrfnj4xPv_C8cT1E3ZQs#1


## Web Application:
http://ec2-18-232-35-95.compute-1.amazonaws.com:8501/

## Architecture 

![](https://github.com/jayshilj/Team3_CSYE7245_Spring2021/blob/main/Final%20Project/Architecture%20Final%20AWS_FAST.jpg)

## Project Structure
```
Project
C:.
|  audioapp.py
|  googletrends.py
|  jayshilapp.py
|  output.doc
|  README.md
|  stockapp-FinViz.py
|  StockScreenDemo.py
|  summary.py
|  tweetData.py
|  
+---Datasets
|      SP500.csv
|      
+---Fastwebapp
|  |  app.py
|  |  LICENSE.txt
|  |  Procfile
|  |  README.md
|  |  requirements.txt
|  |  runtime.txt
|  |  
|  +---static
|  |  |  architecture.gif
|  |  |  automobile.gif
|  |  |  bigdata.gif
|  |  |  cloud.gif
|  |  |  data.gif
|  |  |  electronics.gif
|  |  |  Final_AWS_FAST.jpg
|  |  |  forecasting.jpg
|  |  |  medical.gif
|  |  |  retail.gif
|  |  |  sentiment.gif
|  |  |  sentiment_analysis.gif
|  |  |  sentiment_analysis.png
|  |  |  stock.gif
|  |  |  stockmarket.gif
|  |  |  stock_latest.gif
|  |  |  stock_latest.jpg
|  |  |  timeforecasting.gif
|  |  |  workflow_png.png
|  |  |  
|  |  +---css
|  |  |      font-awesome.min.css
|  |  |      main.css
|  |  |      
|  |  +---css_1
|  |  |  |  font-awesome.min.css
|  |  |  |  main.css
|  |  |  |  
|  |  |  \---_notes
|  |  |          dwsync.xml
|  |  |          
|  |  +---fonts
|  |  |      fontawesome-webfont.eot
|  |  |      fontawesome-webfont.svg
|  |  |      fontawesome-webfont.ttf
|  |  |      fontawesome-webfont.woff
|  |  |      fontawesome-webfont.woff2
|  |  |      FontAwesome.otf
|  |  |      
|  |  +---fonts_1
|  |  |  |  fontawesome-webfont.eot
|  |  |  |  fontawesome-webfont.svg
|  |  |  |  fontawesome-webfont.ttf
|  |  |  |  fontawesome-webfont.woff
|  |  |  |  fontawesome-webfont.woff2
|  |  |  |  FontAwesome.otf
|  |  |  |  
|  |  |  \---_notes
|  |  |          dwsync.xml
|  |  |          
|  |  +---js
|  |  |      jquery.min.js
|  |  |      jquery.scrollex.min.js
|  |  |      main.js
|  |  |       skel.min.js
|  |  |      util.js
|  |  |      
|  |  \---js_1
|  |      |  jquery.min.js
|  |      |  jquery.scrolly.min.js
|  |      |  main.js
|  |      |  skel.min.js
|  |      |  util.js
|  |      |  
|  |      \---_notes
|  |              dwsync.xml
|  |              
|  \---templates
|          architecture.html
|          automobile.html
|          cloud.html
|          dataeda.html
|          electronics.html
|          health.html
|          index.html
|          metric.html
|          retail.html
|          sentimentanalysis.html
|          timeseriesanalysis.html
|          
+---GLUE ETL SCRIPTS
|      ETLMigrationJob
|      google_trends_automobile
|      google_trends_cloud
|      google_trends_hardware
|      google_trends_healthcare
|      google_trends_retail
|      sp500
|      TwitterStreamsETL
|      
\---Twitter
       App_Streamlit.py
       Logo1.jpg
```


## Getting Started

#### Prerequisites
1. Python3.7+
2. Docker
3. Flask
4. AWS
5. Streamlit
6. Weights and Biases
7. Twitter Developer Account

#### Configuring the AWS CLI
You need to retrieve AWS credentials that allow your AWS CLI to access AWS resources.

1. Sign into the AWS console. This simply requires that you sign in with the email and password you used to create your account. If you already have an AWS account, be sure to log in as the root user.
2. Choose your account name in the navigation bar at the top right, and then choose My Security Credentials.
3. Expand the Access keys (access key ID and secret access key) section.
4. Press Create New Access Key.
5. Press Download Key File to download a CSV file that contains your new AccessKeyId and SecretKey. Keep this file somewhere where you can find it easily
6. Get AWS Key and create a config file
7. Setup awscli on your local machine 

#### Configuring the Twitter Developer Account
1. Go to https://developer.twitter.com/ and get key to retrive the twiiter data and paste it in a config file.
2. Request access to customer key, consumer key, secrect consumer key and secrect customer key
3. Change the Keys in the app.py file

#### Steps to get the Data
1. git clone the repo https://github.com/jayshilj/Team3_CSYE7245_Spring2021/FinalProject
2. In "Data" folder we have file to run the api and the Scrapper function. This is also scheduled with AWS Lambda in AWS console to run daily and can be modified as per the need.
3. This will get us the data in S3 bucket.
4. Now, We will have a Data in S3 bucket. Now use the AWS glue scripts to build Glue jobs to extract data from S3 buckets, transform it and load it into the Redshift Data Warehouse.

#### Aws Comprehend:
1. In this repo we have python script for sentiment_analaysis we need to run that in order to get sentiment score of the scrapped data which will trigger the aws gule workflow to run the gule jobs which add the data in redshift data warehouse.

#### AWS Lambda Setup:
1. Use the Scrapper Files in the Document to setup the Lambda Function in AWS
2. The Lambda function has to be named in the way the file is named

#### Deploying the Webapp on EC2 instance :
1. Download heroku toolbelt from  https://toolbelt.heroku.com/
2. Creating requirements.txt in which the dependencies for the package are listed on each line in the same folder as app.py. We can list the following:
Flask,
gunicorn
3. Creating runtime.txt which tells Heroku which Python version to use. We have used python-3.5.1
4. Create a Procfile. It is a text file in the root directory of the application that defines process types and explicitly declares what command should be executed to start our app. It can contain:
web: gunicorn app:app --log-file=-
5. We need to create a GitHub repository with app.py and these essential files along with.gitignore(Although it is not necessary it is recommended to include it)
6. Now our Flask app folder contains the this file structure
```
 ├── .gitignore
 ├── Procfile
 ├── app.py
 ├── requirements.txt
 │── runtime.txt
 ```
7. Go on Heroku website and after logging in click on New → Create New App.
Enter ”App name” and select the region and click on Create App and in the Deploy tab, select GitHub in Deployment method then Select the repository name and click Connect
8. Select Enable Automatic Deploys so that whenever we push changes to our GitHub it reflects on our app and Click Deploy Branch  which will deploy the current state of the branch.
If everything is successful it shows an Open App button. We can now open the app deployed on Heroku


#### Docker setup for app:
1. git clone the repo https://github.com/jayshilj/Team3_CSYE7245_Spring2021/FinalProject
2. docker build -t stock_app:1.0 . -- this references the Dockerfile at . (current directory) to build our Docker image & tags the docker image with stock_app:1.0
3. Run docker images & find the image id of the newly built Docker image, OR run docker images | grep stock_app:1.0 | awk '{print $3}'
4. docker run -it --rm -p 5000:5000 {image_id} stock_app:1.0 -- this refers to the image we built to run a Docker container.

### Tests:
1. AMAZON EC2 - Once App is deployed, you can spin the app from your browser, to see if its working or not.
2. Docker- You test it on 0.0.0.0:5000 or using docker-machine ip (eg : http://192.168.99.100:5000/)

### Stage 1:  WEB SCRAPING, DATA PREPROCESSING, LABELLING AUTOMATED PIPELINE
Getting various Company Data from World resources for use case Financial Analytics on Trends and for use case 2 ( text summarization & sentimental analysis on audio files) we will feed on our self generated dataset from Amazon Polly 
Historical call audio data will be into S3 Bucket which will be fed into Database 
To feed realistic trends we will feed on real time tweets from twitter, google trends and various news sources to get headlines 
Cloudwatch is used to monitor PUT operations on the S3 bucket and invoke a Step function
To run Lambda to get all raw data into S3 Bucket
For example : Step Function runs following : 
Scraping from Twitter feed streams into S3 Bucket 
Scraping from Financial News streams into S3 Bucket
Scrapping from Yahoo Stocks into S3 Bucket

###  Stage 2 :  AWS EXTRACT TRANSFORM LOAD PIPELINE 
AWS Glue scheduled crawlers in every hour to crawl and fetch data from various sources 
AWS Catalog will serve as Metadata database for all catalog metadata
AWS Glue Jobs are then used to function on Crawled data which is fed into Redshift 
All Extract Transform load Scripts are stored in S3 and final data is into Redshift
We have leverage Redshift ML query auto scheduler to Truncate our old historical data

###  Stage 3 : FINANCIAL MEETINGS AUDIO CALL TRANSCRIPTION, TEXT SUMMARIZATION 
Defines all Process for transcribed audio to text and  getting text summarization & Sentimental Analysis on Financial Audio Calls
Uses Amazon Transcribe to convert Audio to Text and uses Text Summarization Gensim algorithm to give summarization of audio files and also generate for the same

###  Stage 4 :  FORECASTING PIPELINE 
Streamlit will be running on the EC2 instance, will provide various functionality of forecasting Stock prices & Trends thus allowing business team to make decisions based on sentiments of forecasted prices of stocks for targeted company 

### Stage 5 : STREAMING TWEETS PIPELINE , LIVE NEWS & GOOGLE TRENDS PIPELINE
Streamlit will be running on the EC2 instance, will provide various functionality of fetching Tweets, and perform various algorithms via functionality like word cloud formation of negative sentiments, positive sentiments, etc.

###  Stage 6:  USER AUTHENTICATION PIPELINE
Streamlit will be running on the EC2 instance, will provide various functionality, In this stage, the Lambda function is invoked  via APIs to get Token from cognito and email is sent on user’s email id as an MFA and data is fed into Dynamo DB

###  Stage 7:  MASKING ENTITY PIPELINE
Streamlit will be running on the EC2 instance, will provide various functionality, In this stage, the Lambda function is invoked  via APIs to get done masking from S3 files to comprehend jobs and get data into s3 and then to streamlit

### Stage 8:  INTERACTIVE DASHBOARD WEB APPLICATION PIPELINE 
FLASK web app integrated Dashboard will be running on the EC2 instance, will provide various functionality In this stage 

###  Stage 9: TECHNICAL DOCUMENTATION 
Collated results are sent for report generation through the Reporting tool
Amazon Simple Notification Service will be used for any pipeline failures, successful runs, and triggering email notifications.


## Authors
<b>[Akash M Dubey](https://www.linkedin.com/in/akashmdubey/)</b> 

<b>[Sagar Shah](https://www.linkedin.com/in/shahsagar95/)</b> 

<b>[Jayshil Jain](https://www.linkedin.com/in/jayshiljain/)</b> 

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
