# Spotify Billboard Hot 100 Hits Prediction

## Overview

This project predicts which songs are likely to become Billboard Hot 100 hits using Spotify track data and machine learning techniques. The aim is to identify key factors contributing to a song's success and provide actionable insights for music industry professionals.

## Motivation

The Billboard Hot 100 is a globally recognized standard for measuring a song's popularity. This project was developed to:

1. Identify the attributes influencing chart success.
2. Equip artists, producers, and marketers with data-driven insights for optimizing their music.

## Dataset Description

### Data Source

**Spotify Tracks Dataset**

- **Total Records:** 114,000 (reduced to 71,945 after preprocessing).
- **Attributes:**
  - **Categorical Features:** 5.
  - **Numerical Features:** 15 (+1 derived feature).

### Preprocessing Steps

1. **Cleaning:**
   - Removed null rows.
   - Dropped unnamed index columns.
   - Deduplicated rows using combinations of key columns.
   - Converted categorical features (e.g., "explicit") to numerical values.
2. **Filtering:**
   - Removed rows where:
     - `speechiness > 0.7` (likely podcasts).
     - `instrumentalness > 0.8` (instrumental remakes).
     - `liveness > 0.8` (live streams).
     - `duration_ms = 0`.
3. **Dimensionality Reduction:**
   - Applied PCA to reduce dimensions to 13 columns while retaining 95% variance.
   - Applied target encoding for the "genre" attribute.

## Methodology

The project follows a structured pipeline:

1. **Regression model:**
   - We trained lasso, ridge, elasticnet, random forest, ADA Boost K Neighours XGBoost.
   - Accuracy: ~40%.
2. **Binary Classification:**
   - Categorized songs into "hit" or "non-hit" classes.
   - Random Forest achieved ~96% accuracy.
3. **Overfitted Regression Model:**
   - Used songs with popularity > 70 to predict numerical popularity scores
   - Applied regression on high-popularity songs to predict fine-grained popularity scores.
   - Accuracy: ~20%.

### 1. Building the Overfitted Regression Model

- **Purpose**: To overfit the training dataset and capture all patterns, including noise.
- **Method**:
  - Used the entire dataset and all features with minimal regularization.
  - Focused on maximizing regression accuracy on the training data.
- **Outcome**: Created a regression model highly tuned to the training data for precise numerical score predictions.

### 2. Training the Classification Model

- **Purpose**: To distinguish high-popularity songs (potential hits) from non-hits.
- **Method**:
  - Trained a classification model (e.g., Random Forest) on a filtered subset of the dataset where the popularity score > 70.
  - Targeted high-popularity songs likely to enter the Billboard Top 100.
- **Outcome**: A model capable of filtering songs based on hit likelihood.

### 3. Filtering Data Using Classification Predictions

- **Purpose**: To narrow down the dataset to high-potential songs.
- **Method**:
  - Applied the classification model to predict the likelihood of songs being hits.
  - Retained only songs with high predicted popularity scores.
- **Outcome**: A reduced dataset of strong Billboard Top 100 contenders.

### 4. Final Regression and Ranking

- **Purpose**: To assign fine-grained popularity scores and rank songs.
- **Method**:
  - Input the filtered data into the overfitted regression model.
  - Predicted numerical popularity scores for these high-potential songs.
  - Ranked songs in descending order of predicted scores.
- **Outcome**: A ranked list of songs based on their likelihood of chart success.

## Results

Key insights from the project:

- **Top Attributes for Chart Success:** Popularity, danceability, energy, loudness, and valence.
- **Best Model:** Random Forest for binary classification with ~96% accuracy.
- Regression models performed moderately but struggled with overfitted datasets.

## Technologies and Libraries

- **Programming Language:** Python
- **Libraries:**
  - Data Processing: Pandas, NumPy
  - Machine Learning: scikit-learn
  - Visualization: Matplotlib, Seaborn

## Project Workflow

1. **Dataset Preparation:**
   - Input and preprocess Spotify Tracks Dataset.
2. **Regression Model:**
   - Train regression models on songs with popularity > 70.
   - Predict numerical popularity scores.
3. **Binary Classification:**
   - Train classification models to identify hits.
4. **Overfitted Regression Model:**
   - Test predictions on filtered, high-popularity songs.
   - Rank songs based on predicted popularity scores.
5. **Combined Pipeline:**
   - Integrate regression and classification models to identify top hits.

## How to Run the Code

### Files and Order of Execution

1. **Dataset:** Place your Spotify Tracks Dataset (`spotify_tracks_dataset.csv`) in the root folder.
2. **Preprocessing:** Run `data_processing.ipynb` to clean and preprocess the dataset.
3. **Classification Model:** Run `classification_model.ipynb` to train the binary classification model.
4. **Regression Model:**
   - Run `regression_model.ipynb` to train the initial regression model.
   - Run `regression_70to100_model.ipynb` to train the regression model on high-popularity songs.
5. **Combined Pipeline:** Run `combined_pipeline.ipynb` to execute the full workflow and generate predictions.

## Authors

- **Rohit Agrawal(24M0744)**
- **Daksh Goyal(24M0756)**
- **Gaurav Kumar(24M0786)**
- **Anishek Chaudhary(24M0787)**

## Future Work

- Improve regression model accuracy.
- Incorporate additional features from other platforms (e.g., Apple Music, YouTube).
- Further optimize classification models for real-world performance.

## License

This project is open-source under the MIT License.
