# Churn Prediction and Comparative Analysis

## Overview

This project focuses on predicting customer churn using K-Means Clustering for customer segmentation and evaluating the performance of various classification models, including Artificial Neural Networks (ANN), Random Forest, and Support Vector Machines (SVM). Churn prediction is a critical task in customer retention and business management, and this project aims to provide valuable insights and tools for decision-making.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- Python prerequisites required for data preprocessing, predicting churn and for the evaluation step are mentioned in "requirements.txt" file.
## Methodology
The following variables/attributes are available within the dataset:
- ID
- userID
- user account creation date
- Number of pages viewed by the user
- View duration
- ID of the brochure viewed
- app installation date
- ID of the product viewed
- model of the product viewed
- ID of the campaign

In order to predict churn, the following steps are implemented:
- Data preprocessing
- Exploratory data analysis (EDA)
- Predicting Churn
- Validation

### Data preprocessing
The preprocessing step is used to remove any abnormalities existing in the data, correct the date/time format and append all the data into a single pandas dataframe.

### Exploratory data analysis (EDA)
- This step is used to initially summarize the data.
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histograms_relevant_variables.png)

- Next step is to discard variables irrelevant in predicting churn, such as:
    - ID
    - userID
    - brochure ID
    - product ID
    - model
    - campaign ID

### Predicting churn
An unsupervised machine learning is utilized to calculate and predict churn. This is implemented using KMeans clustering. The correct number of clusters to used is recognized using the elbow method.
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Elbow_method_for_kmeans.png)
Based on the elbow method, Churn is classified into three types.
- High risk
- Mid risk
- low risk
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histogram_churn_count.png)
Further data analysis is done using histogram to visualy identify the soundness of KMeans churn prediction.
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histograms_date_created.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histograms_install_date.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histograms_page_turn_count.png)

![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Churn_prediction_and_comparative_analysis/plots/Histograms_view_duration.png)

## Validation using classification models

The results from KMeans with respect to churn prediction are further validated using the following models.
- Artificial neural network
- Random forest classifier
- Support vector machine classifier

The classification reports as well as the confusion matrix generated from these models can be seen in "evaluation_results" folder.