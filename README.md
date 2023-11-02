
# Disaster Response Pipeline Project
### Summary of project

In this project, I applied data engineering skills learnt in the course to analyze disaster data from Appen (formally Figure 8) and built a model for an API that classifies disaster messages.

### File descriptions
README.md
Folder: app
	 run.py # Flask file that runs app
	 Folder: template
    	 go.html # classification result page of web app
    	 master.html # main page of web app
         
Folder: data
	 DisasterResponse.db # database saved from process_data
	 disaster_categories.csv # data to process
	 disaster_messages.csv # data to process
	 process_data.py # data cleaning pipeline
     
Folder: models
	 train_classifier.py # machine learning pipeline

classifier.pkl # model saved from train_classifier

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
