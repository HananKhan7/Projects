"""
Project: Text Summarization using OpenAI GPT 3.5 Turbo

Description:
This project generates summaries from survey responses using OpenAI’s API. The summaries aim to address the problems mentioned in the responses, highlighting the most repeated issues based on their frequency.

Main steps:
1. Set up OpenAI API credentials.
2. Load the data.
3. Filter data based on specific criteria (years, series, and brands).
4. Preprocess the data.
5. Group and summarize the comments.
6. Generate summaries using OpenAI’s API.
7. Format the results.
8. Save the summarized results into a CSV file.

Requirements:
- Python 3.6 or higher
- Required Python libraries (specified in the imports section)
- OpenAI API credentials
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import openai
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import nltk
import re
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Download necessary NLTK data
nltk.download('popular')

# DataFrame compatibility fix for older Pandas versions
pd.DataFrame.iteritems = pd.DataFrame.items

# Set up OpenAI API credentials
client = AzureOpenAI(
    azure_endpoint="https://agco-can-sandbox.openai.azure.com/",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-05-01-preview"
)

# Parameters and Variables
path = 's3://dev-agco-datastewards-eme/datalake/analytics/scoring_dataset/dump/Farmer_insights_q2_extract.csv'
years = ['2024-1', '2024-2', '2024-3', '2024-4', '2024-5', '2024-6']
series = [
    '4700 Global Srs. Tractor', '4700M Srs. Tractor', 
    '5700 Dyna-4 Srs. Tractor', '5700 Global Srs. Tractor', 
    '5700M Srs. Tractors', '700 Gen6 (Tier 5)', '700 Gen7'
]
brands = ['Massey Ferguson', 'Fendt']

# Dimensionality Reduction
cols_dim_reduction = [
    'Brand', 'Global Machine Series', 'Global Machine Model', 'Topic', 
    'Subtopic', 'Element', 'Comment Phrase', 'Translated Full Comment', 
    'Sentiment'
]
groupby_cols = [
    'Brand', 'Global Machine Series', 'Global Machine Model', 'Topic', 
    'Subtopic', 'Element'
]

# Functions

def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stop words, and lemmatizing.
    
    Args:
        text (str): The text to be preprocessed.
        
    Returns:
        str: The preprocessed text.
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = nltk.tokenize.word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

def generate_summary(reviews, category):
    """
    Generate a summary for the given text using OpenAI's API.
    
    Args:
        phrases (str): The text to be summarized.
        category (str): The category of the text.
        
    Returns:
        str: The generated summary.
    """

    messages = [
    {"role": "system", "content": """You are an expert in analyzing product reviews. You will be provided with product reviews related to a specific product category. Each sentence in the provided text represents a different review, and these sentences are aggregated together in a single text, seperated by periods. Your task is to summarize the problems mentioned in these reviews, focusing only on those that are relevant to the product category. Please follow the instructions below:

    1. Summarize the problems relevant to the product category: Include only sentences that are relevant to the product category.

    2. Identify repeated problems: Pay special attention to problems that are mentioned in multiple distinct reviews. 

    3. Ignore irrelevant information: If the provided text does not contain relevant information, return only "N/A".

    Use the following structure for your response:

    1. If there are relevant problems:
    1.1. The problems mentioned in the reviews are <...>. The most common issues reported are <...>.
    2. If there is only one relevant problem:
    2.1. The problem mentioned in the review is <...>.
        
    """},

    {"role": "assistant", "content": """The problems mentioned in the reviews are <...>. The most common issues reported are <...>."""},
    {"role": "assistant", "content": """The problems mentioned in the reviews are <...>. """},
    {"role": "assistant", "content": """N/A """},
    {"role": "assistant", "content": """The problem mentioned in the review is <...>. """},

    {"role": "user", "content": f""""Please summarize the problems relevant to Product Category up to 2 sentences. Please pay special attention to the repeated problems. No need to write Product Category in your response. 

    Product Category: ""{category}""
    Reviews: "{reviews}""
    
    """}
] 
    completion_negative_imp = client.chat.completions.create(
        model="gpt35-turbo",
        messages=messages,
        max_tokens=1500, #3/4 words is 1 token
        n=1,
        stop=None,
        temperature=0.0

    )
    response = completion_negative_imp.choices[0].message.content
    time.sleep(0.2)

    return response

def format_results(results):
    """
    Format the results into the required dataframes.
    
    Args:
        results (list): The list of results to be formatted.
        
    Returns:
        tuple: A tuple containing the subtopic level dataframe, element level dataframe with summary, and table view dataframe.
    """
    subtopic_df = pd.DataFrame(results)
    element_df = pd.DataFrame(results)
    table_view_df = pd.DataFrame(results)
    return subtopic_df, element_df, table_view_df

def rearrange_text(text):
    """
    Rearrange text to ensure sentences starting with "The most" appear first.
    
    Args:
        text (str): The text to be rearranged.
        
    Returns:
        str: The rearranged text.
    """
    sentences = text.split('. ')
    first_sentence = [s for s in sentences if s.startswith("The most")]
    other_sentences = [s for s in sentences if not s.startswith("The most")]
    return ' '.join(first_sentence + other_sentences) if first_sentence else text

def replace_text(formatting):
    """
    Replace specific text patterns in the given string.
    
    Args:
        formatting (str): The text to be formatted.
        
    Returns:
        str: The formatted text.
    """
    formatting = formatting.replace('1.', '').replace('2.', '').replace('3.', '')
    if "The most" in formatting:
        return formatting.replace("The problems", "All problems")
    elif 'The provided text does not contain' in formatting:
        return '-'
    elif 'sorry, but the provided text' in formatting:
        return '-'
    elif 'N/A' in formatting:
        return '-'
    else:
        return formatting

def create_transmission(value):
    """
    Create a transmission type based on the given value.
    
    Args:
        value (str): The value to check for transmission types.
        
    Returns:
        str: The transmission type.
    """
    if "Dyna7" in value:
        return "Dyna7"
    elif "DynaVT" in value:
        return "DynaVT"
    elif "Dyna E-Power" in value:
        return "Dyna E-Power"
    else:
        return 'N/A'

def eliminate_general_element(text):
    """
    Check if the text contains "General" or "general".
    
    Args:
        text (str): The text to be checked.
        
    Returns:
        int: 1 if "General" or "general" is found, otherwise 0.
    """
    return 1 if "General" in text or "general" in text else 0

def main():
    """
    Main function to execute the text summarization workflow.
    """
    # Load and preprocess data
    data = pd.read_csv(path)
    
    # Filter data based on years, series, and brands
    data = data[data['Survey_YearMonth'].isin(years)]
    data = data[data['Global_Machine_Series'].isin(series)]
    data = data[data['Brand'].isin(brands)]
    
    # Dimensionality reduction
    data = data[cols_dim_reduction]
    
    # Example of processing and summarizing text
    data['processed_text'] = data['Comment Phrase'].apply(preprocess_text)
    
    # Generate summaries
    grouped_df_negative = data.groupby(['Brand', 'Global Machine Series', 'Global Machine Model', 'Topic', 'Subtopic', 'Element'])['Comment Phrase'].apply(lambda x: '. '.join(x)).reset_index()
    grouped_df_negative['comment_count'] = grouped_df_negative['Comment Phrase'].apply(lambda x: len(x.split('. ')))
    
    df_with_summary = grouped_df_negative[grouped_df_negative["comment_count"] > 1].copy()
    
    for index, row in tqdm(df_with_summary.iterrows(), desc='Processing Summaries', total=df_with_summary.shape[0]):
        phrases = row['Comment Phrase']
        category = row['Element']
        summary = generate_summary(phrases, category)
        df_with_summary.loc[index, 'Summary'] = summary
    
    # Format results
    df_with_summary['sentence_order'] = df_with_summary['Summary'].apply(rearrange_text)
    df_with_summary['replace_text'] = df_with_summary['sentence_order'].apply(replace_text)
    
    # Save summarized results
    df_with_summary.to_csv('summary_results.csv', index=False)
    print("Summarization completed and results saved.")

if __name__ == "__main__":
    main()
