
# Text Summarization of Customer Responses using GPT 3.5 Turbo

This repository contains a Python script that uses OpenAI’s GPT 3.5 Turbo model to generate summaries of customer responses, highlighting the most frequently mentioned problems. The goal of this project is to streamline the analysis of survey responses by providing concise summaries that emphasize recurring issues.

## Project Description

This project leverages the GPT 3.5 Turbo model to summarize customer feedback from surveys. The script processes customer responses, generates summaries that highlight the most mentioned problems, and saves the summarized results for further analysis.

### Objectives

- **Summarize Customer Feedback**: Generate concise summaries of customer responses to identify frequently mentioned problems.
- **Model Implementation**: Utilize the GPT 3.5 Turbo model for text summarization.
- **Highlight Recurring Issues**: Emphasize the most frequently mentioned problems in the summaries.

## Files in the Repository

1. **Text_summarization_GPT.py**: This script uses the GPT 3.5 Turbo model to summarize customer responses and highlight frequently mentioned problems.

## Steps Taken

### Data Preparation

1. **Load Environment Variables**: Set up API credentials using environment variables.
2. **Load Data**: Load survey response data from a specified path.
3. **Filter Data**: Filter the data based on specific criteria (years, series, and brands).

### Text Processing

1. **Preprocess Text**: Tokenize, remove stop words, and clean the text data.
2. **Group Comments**: Group comments by specific categories for summarization.

### Model Implementation

1. **Load Model**: Set up and authenticate the GPT 3.5 Turbo model using OpenAI’s API.
2. **Generate Summaries**: Generate summaries for grouped comments, emphasizing frequently mentioned problems.

### Results Handling

1. **Format Results**: Format the generated summaries for readability.
2. **Save Results**: Save the summarized results into a CSV file for further analysis.

## How to Use

### Prerequisites

- Python 3.6 or higher
- Required Python libraries (specified in the imports section of the script)
- OpenAI API credentials

### Installation

1. Install the required libraries:

### Running the Script

To run the script, execute the following command:
```bash
python Text_summarization_GPT.py
```

## Results

The output will be a CSV file containing the summarized customer responses, highlighting the most frequently mentioned problems. The results can be used to analyze customer feedback effectively and identify common issues.
