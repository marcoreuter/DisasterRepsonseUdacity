
# DisasterRepsonseUdacity

## Github repository for the project "Disaster Response Pipeline" of the Udacity Data Scientist Nanodegree

### Business Understanding
The project analyzes messages that are sent by individuals who are affected by disasters. It uses machine learning techniques in order to categorize them into relevant categories. This categorisation can be used by Disaster relief organizations to efficiently respond to the needs of those affected by the disaster.

### Data Understanding
The raw data is contained in 2 .csv files. The messages.csv file contains the following columns:

- Id: A unique ID for each message
- message: an english translation of the message (if applicable)
- original: the original text of the message
- genre: the type of message, for example a direct message or a news article

![Example message](messages_example.png)

The categories.csv file contains the following columns:
- Id: A unique ID for each messages 
- categories: a string that contains a pre existing classification of the message into on or more of the 36 categories
   
![A test image](categories_example.png)

### Data Preparation
The single string of the "categories" column is split up into 36 columns that either contain a "1" or "0" decoding whether or not the message belongs to that category. Then a single dataframe is created. Some data cleaning is necessary, in particular:

- The data set contains some duplicates. They have been removed.
- The category classification contains some "2" values, which are likely typos. They have been removed.

The cleaned dataframe is stored in a SQL table for later use. This process is automated into an ETL pipeline.

### Modeling
The raw data is in form of natural language. It is processed using the following data pipeline:

1) Feature Creation/Transformation:
- Text is tokenized and lemmatized using the NLTK library
- Text is analyzed using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

2) Prediction
 - A model is trained and evaluated in a test-train-split
 - Different models are used and evaluated according to their f1 scores, precision and recall
 - Crossvalidation is used to select the final model / paremeters. The canidate models include
	 - Random forest
	 - K-nearest Neighbours

This ML pipeline is stored and the final model is exported.
##### The model training may take a long time to complete through the amount of folds in the cross validation and the size of the dataset!

### Evaluation
The final model that performed best is a RandomForest Classifier. The best model was selected according to a weighted f1scores over the 36 categories that are to be classified. The weighted f1score of the best performing model is 0.57
### Deployment
The final model from the modelling section is used to deploy a web app using Python as a back end. Flask, json and plotly are used to connect and create a front end. Several template for this are provided by Udacity and used in the project.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pklcls`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


#### File Descriptions

#### Root Directory:
README.MD: Readme file  
create_wordcloud.py: Python script to create the wordcloud image for the website  
categories_example.png: Example picture for categories data  
messages_example.png: Example picture for messages data  

##### app directory:
run.py: python script to deploy the website through flask
###### app/static:
wordcloud.png: wordcloud picture for the website
###### app/templates
go.html: template created by udacity to use in the project  
master.html: templace created by udacity to use in the project
##### data directory:
disaster_categories.csv: .csv file with categories  
disaster_messages: .csv file with messages  
DisasterResponse.db: SQL database with cleaned data  
process_data.py: python script to load and clean data  

##### models directory
train_classifier.py: python script to train and save model  
classifier.pklcls: best model, careful, large file  
.gitattributes: needed to score large file in github  

#### Packages and data sources
1) Packages used in the project:
- Pandas
- Numpy
- SQLAlchemy
- NLTK
- sklearn
- json
- plotly
- flask

2) Data provided by [Udacity](https://www.udacity.com) as a part of the Data Scientist Nanodegree in coorperation with [Figure Eight](https://figure-eight.com).


