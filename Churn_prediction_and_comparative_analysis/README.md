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


## Dataset

Describe the dataset used in the project. Include information about its source, structure, and any preprocessing steps performed on the data. You can also provide a link to the dataset if it's publicly available.

## Installation

Explain how to set up the project environment and install any necessary dependencies. You can include code snippets or a step-by-step guide to help users get started.

```shell
# Example installation steps
git clone https://github.com/yourusername/churn-prediction-project.git
cd churn-prediction-project
pip install -r requirements.txt