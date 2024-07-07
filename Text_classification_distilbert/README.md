
# Customer Comments Topic Classification using DistilBERT

This repository contains a Python script that implements a fine-tuned DistilBERT model for classifying customer responses into predefined topics. The goal of this project is to enhance the assessment of customer feedback by accurately categorizing comments, enabling better analysis and insights.

## Project Description

This project utilizes a fine-tuned DistilBERT model to classify customer comments into predefined topics. The script processes customer feedback, predicts topics, and applies a threshold criterion to filter out low-scoring predictions, ensuring the presence of important keywords.

### Objectives

- **Customer Feedback Classification**: Classify customer comments into specific topics to improve the assessment of customer feedback.
- **Model Implementation**: Implement a fine-tuned DistilBERT model for sequence classification.
- **Prediction Filtering**: Apply thresholds to filter out low-confidence predictions.

## Files in the Repository

1. **BERT_classification_customer_comments.py**: This script uses the fine-tuned DistilBERT model to classify customer comments into predefined topics.

## Steps Taken

### Data Preparation

1. **Preprocess Comments**: Tokenize, remove stop words, and lemmatize the customer comments.
2. **Load Configuration**: Load necessary configurations from YAML files.

### Model Implementation

1. **Load Model and Tokenizer**: Load the pre-trained DistilBERT model and tokenizer for sequence classification.
2. **Generate Predictions**: Predict topics for each customer comment using the model.

### Prediction and Evaluation

1. **Filter Predictions**: Filter the predictions based on a specified threshold to ensure high-quality topic classification.
2. **Save Results**: Save the prediction results to a CSV file for further analysis.

## How to Use

### Prerequisites

- Python 3.6 or higher
- Required Python libraries (specified in the imports section of the script)
- Pre-trained DistilBERT model and tokenizer
- Configuration files: `config.yaml` and `dynamic_config.yaml`

### Installation

1. Install the required libraries:


### Running the Script

To run the script, execute the following command:
```bash
python BERT_classification_customer_comments.py
```

## Results

The output will be a CSV file containing the classified topics for each customer comment. The results can be used to analyze the effectiveness of the topic classification and to gain insights into customer feedback.
