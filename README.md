# Disaster_Response_Pipelines


# Introduction
After a natural disaster, thousands of people send out messages to ask for help through various channels such as social media. Like that they need food; or they're trapped under rubble. However, the government does not have enough time to read all the messages and send them to various departments. Then, this project will play an important role and help people to be safe.<br>
This Project is required as a part of the Data Science Nanodegree Program of Udacity in collaboration with appen. The initial dataset contains pre-labelled tweet and messages from real-life disaster situations. The aim of the project is to build a Natural Language Processing tool that categorize messages.

### The Project is divided in the following Sections:

- Processing Data, ETL Pipeline for extracting data from source, cleaning data and saving them in a proper database structure.
- Machine Learning Pipeline for training a model to be able to classify text message in categories.
- Web App for showing model results in real time.

# Files and folders

- data: folder contains sample messages and categories datasets in csv format<br>
|- disaster_categories.csv # data to process<br>
|- disaster_messages.csv # data to process<br>
|- process_data.py # python code takes as input csv files(message data and message categories datasets), clean it, and then creates a SQL database <br>
|- InsertDatabaseName.db # database to save clean data to <br>
- app: contains the run.py to deploy the web app.<br>
| - template<br>
| |- master.html # main page of web app<br>
| |- go.html # classification result page of web app<br>
|- run.py # Flask file that runs app<br>
- models<br>
|- train_classifier.py # code to train the ML model with the SQL data base<br>
|- classifier.pkl # saved model<br>
|- ETL Pipeline Preparation.ipynb: process_data.py development process<br>
|- ML Pipeline Preparation.ipynb: train_classifier.py development process <br>
- README.md<br>




# Installation
python (=>3.6) <br>
pandas
<br>numpy
<br>sqlalchemy
<br>sys
<br>plotly
<br>sklearn
<br>joblib
<br>flask
<br>nltk

## Instructions:
1- Run the following commands in the project's root directory to set up your database and model.

2- To run ETL pipeline that cleans data and stores in database ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```

3- To run ML pipeline that trains classifier and saves ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

4- Run the following command in the app's directory to run your web app. ```python run.py```

5- Go to http://0.0.0.0:3001/


## Acknowledgements
- Udacity for providing an amazing Data Science Nanodegree Program
- Figure Eight for providing the relevant dataset to train the model

