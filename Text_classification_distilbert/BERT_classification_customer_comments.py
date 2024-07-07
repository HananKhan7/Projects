"""
Project: Customer Comments Topic Classification using distilBERT

Description:
This project aims to classify customer comments into predefined topics using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) distilled model.
The script processes customer feedback, predicts topics, and applies a threshold criterion to filter out low-scoring predictions and ensure the presence of important keywords.

The prediction can be done in two ways:
- Full version: Gather data based on the delta date and overwrite the existing data.
- Delta version: Gather data based on the delta date and append to the existing data.

Main Features:
1. Preprocessing of customer comments including tokenization, stop word removal, and lemmatization.
2. Loading of a pre-trained BERT model and tokenizer for sequence classification.
3. Prediction of topics for each customer comment.
4. Filtering of predictions based on a specified threshold to ensure high-quality topic classification.
5. Saving of the prediction results to a CSV file for further analysis.

Usage:
To run the script, simply execute it. The main function will demonstrate how to preprocess comments, load the model and tokenizer, predict topics, and save the results.

Requirements:
- Python 3.6 or higher
- Required Python libraries (specified in the imports section)
- Pre-trained BERT model and tokenizer
- Configuration files: config.yaml and dynamic_config.yaml

"""

# Importing libraries and configurations
import os
import ast
import re
import gc
import yaml
import nltk
import openpyxl
import itertools
import pandas as pd
import numpy as np
import s3fs
import mlflow
import mlflow.keras
import torch
import tensorflow as tf
from datetime import datetime, timedelta
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    hamming_loss, confusion_matrix, multilabel_confusion_matrix, 
    ConfusionMatrixDisplay, classification_report
)
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, AUC
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import (
    TFDistilBertForSequenceClassification, DistilBertConfig, 
    DistilBertTokenizerFast, BertConfig, BertModel, BertTokenizer
)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load configuration files
with open('config.yaml') as cfg_file:
    cfg = yaml.safe_load(cfg_file)

with open('dynamic_config.yaml') as dynamic_cfg_file:
    dynamic_cfg = yaml.safe_load(dynamic_cfg_file)

# Define functions
def preprocess_comments(comments):
    """
    Preprocess customer comments by tokenizing, removing stop words, and lemmatizing.
    
    Args:
        comments (list of str): List of customer comments.
        
    Returns:
        list of str: Preprocessed comments.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_comments = []
    for comment in comments:
        tokens = word_tokenize(comment)
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
        processed_comments.append(" ".join(filtered_tokens))

    return processed_comments

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Load a pre-trained BERT model and tokenizer.
    
    Args:
        model_path (str): Path to the pre-trained BERT model.
        tokenizer_path (str): Path to the tokenizer.
        
    Returns:
        model: Loaded BERT model.
        tokenizer: Loaded BERT tokenizer.
    """
    model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    return model, tokenizer

def predict_topics(comments, model, tokenizer, threshold=0.5):
    """
    Predict topics for customer comments using a BERT model.
    
    Args:
        comments (list of str): List of customer comments.
        model: Pre-trained BERT model.
        tokenizer: BERT tokenizer.
        threshold (float): Threshold for filtering low-scoring predictions.
        
    Returns:
        list of dict: Predicted topics with scores.
    """
    inputs = tokenizer(comments, return_tensors='tf', padding=True, truncation=True)
    outputs = model(inputs)
    predictions = tf.nn.sigmoid(outputs.logits).numpy()

    results = []
    for i, prediction in enumerate(predictions):
        topic_scores = {f"topic_{j}": score for j, score in enumerate(prediction) if score >= threshold}
        results.append({"comment": comments[i], "topics": topic_scores})
    
    return results

def save_predictions(predictions, output_path):
    """
    Save predictions to a CSV file.
    
    Args:
        predictions (list of dict): Predicted topics with scores.
        output_path (str): Path to save the CSV file.
    """
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)

def merge_and_filter_data(scoring_dataset, filtered_df, hierarchy_mapping):
    """
    Merge scoring data with filtered data and hierarchy mapping.
    
    Args:
        scoring_dataset (pd.DataFrame): Scoring dataset.
        filtered_df (pd.DataFrame): Filtered dataset.
        hierarchy_mapping (str): Path to hierarchy mapping CSV file.
        
    Returns:
        pd.DataFrame: Combined and filtered dataset.
    """
    df_score_additional_data = pd.merge(
        left=scoring_dataset, right=filtered_df, 
        how='inner', on='scoring_data'
    ).reset_index(drop=True)
    df_score_additional_data.drop_duplicates(subset=['scoring_data', 'topic_id', 'score'], inplace=True)
    
    df_hierarchy_mapping = pd.read_csv(hierarchy_mapping)
    combined_df = pd.merge(
        left=df_score_additional_data, right=df_hierarchy_mapping, 
        how='left', on='topic_id'
    )
    combined_df = combined_df[[
        'agco_id', 'serial_number', 'submit_date', 'region', 'wave', 
        'country_name', 'language', 'brand_name', 'survey', 'data_source',
        'original_comments', 'translated_comments', 'split_sentence', 
        'scoring_data', 'topic', 'subtopic', 'element', 'topic_id', 'score'
    ]]
    combined_df.drop_duplicates(subset=['agco_id', 'serial_number', 'submit_date', 'split_sentence', 'topic_id'], inplace=True)
    combined_df['scoring_data'] = combined_df['scoring_data'].apply(lambda x: str(x) if isinstance(x, float) else x)
    combined_df['scoring_data'] = combined_df['scoring_data'].apply(lambda x: x.lower())
    combined_df.drop_duplicates(subset=['scoring_data', 'topic_id'], inplace=True)
    combined_df.dropna(subset=['split_sentence', 'scoring_data'], inplace=True)
    
    return combined_df

def save_combined_data(combined_df, data_mode, current_date, dynamic_cfg):
    """
    Save combined data to CSV and update dynamic configuration.
    
    Args:
        combined_df (pd.DataFrame): Combined dataset.
        data_mode (str): Data mode (full or delta).
        current_date (str): Current date string.
        dynamic_cfg (dict): Dynamic configuration dictionary.
    """
    combined_df_path = f'/dbfs/mnt/datalake/analytics/scoring_dataset/dump/predicted_data_context_{data_mode}_{current_date}.csv'
    combined_df.to_csv(combined_df_path, index=False)
    
    dynamic_cfg['formatting_params']['predicted_data_file'] = combined_df_path
    
    with open('dynamic_config.yaml', 'w') as d_cfg_file:
        yaml.dump(dynamic_cfg, d_cfg_file)

def main():
    """
    Main function to execute the prediction workflow.
    """
    comments = ["The product is great!", "I am not satisfied with the service."]
    processed_comments = preprocess_comments(comments)
    model, tokenizer = load_model_and_tokenizer(cfg['model_path'], cfg['tokenizer_path'])
    predictions = predict_topics(processed_comments, model, tokenizer, threshold=0.5)
    save_predictions(predictions, 'predicted_topics.csv')

    scoring_dataset = pd.read_csv(cfg['scoring_dataset_path'])
    filtered_df = pd.read_csv(cfg['filtered_dataset_path'])
    hierarchy_mapping = cfg['hierarchy_mapping_path']
    
    combined_df = merge_and_filter_data(scoring_dataset, filtered_df, hierarchy_mapping)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    data_mode = cfg['data_mode']
    
    save_combined_data(combined_df, data_mode, current_date, dynamic_cfg)
    
    print("Predictions and combined data saved.")

if __name__ == "__main__":
    main()
