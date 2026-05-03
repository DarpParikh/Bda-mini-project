# Delhi Crime Analysis Dashboard (PySpark + Streamlit)

## Overview

Delhi Crime Analysis Dashboard is a Big Data Analytics mini project built to explore crime patterns in Delhi using PySpark for data preparation and Streamlit for interactive visualization.

The project processes the raw crime dataset, creates a cleaned and feature-engineered output, and presents a dashboard for crime analysis, filtering, machine learning insights, clustering, prediction, and geospatial visualization.

## Features

- Interactive filters for crime type and hour range
- Data exploration table for browsing filtered records
- Visual analytics for crime trends and top crime types
- Correlation insights for numeric and encoded features
- Crime clustering using KMeans
- Machine learning insights using RandomForest
- Crime prediction system based on location and hour
- Map and heatmap visualization for spatial analysis

## Tech Stack

- Python
- PySpark
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- PyDeck
- Plotly
- Matplotlib
- Seaborn

## Dataset Information

- Source dataset: `crime_dataset_india.csv`
- Processed dataset: `processed_crime_data.csv`
- Main columns used in the dashboard:
  - Crime Description
  - Latitude
  - Longitude
  - Hour

The raw dataset is preprocessed before dashboard analysis so the app can work with a structured and consistent input file.

## How to Run Locally

1. Clone the repository.

   ```bash
   git clone https://github.com/DarpParikh/Bda-mini-project.git
   cd Bda-mini-project
   ```

2. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required packages.

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit dashboard.

   ```bash
   streamlit run app.py
   ```

   To regenerate the processed dataset before launching the dashboard, run:

   ```bash
   python main.py
   ```

## Deployment

The application is deployed on Streamlit Cloud.

- Deployed link: [Add deployed app link here]

## Project Structure

- `app.py` - Main Streamlit dashboard application
- `main.py` - PySpark preprocessing and processed data generation script
- `app_recovered.py` - Helper or recovered application script
- `requirements.txt` - Python dependencies
- `crime_dataset_india.csv` - Raw dataset
- `processed_crime_data.csv` - Preprocessed dataset used by the dashboard

## Machine Learning Explanation

The dashboard uses a RandomForestClassifier to provide crime prediction insights. The model relies on features such as Latitude, Longitude, and Hour, along with engineered time-based and cluster-based features.

Model accuracy is expected to be low because the dataset has limited predictive features. The prediction task is constrained by the available columns, so the model is best understood as an exploratory insight tool rather than a production-grade predictor.

Feature importance helps show which inputs contribute most to the model output. In this project, Latitude, Longitude, and Hour are the primary drivers used for prediction and explanation.

## Limitations

- No real-time data integration
- Limited features for prediction
- Accuracy can be improved with richer location and temporal data
- Clustering and prediction are based on the available dataset only

## Future Improvements

- Add time-series forecasting for crime trends
- Improve the machine learning model with more features
- Add real-time data integration
- Use better geospatial clustering techniques
- Expand the dashboard with more interactive analytics

## Screenshots

### Dashboard Overview

Add screenshot here.

### Visual Analytics

Add screenshot here.

### ML Insights

Add screenshot here.

### Map & Heatmap

Add screenshot here.

## Author

- Name: Khushi Kumari
- Name: Darp Parikh
- Course: B.Tech CSIT, 6th Sem
- University: Reva University
