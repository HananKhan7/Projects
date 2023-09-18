# Importing libraries
import pandas as pd
import re, os
from datetime import datetime

# Initial preprocessing
def data_preprocessing(path:str):
    """
    This method takes in the text path, formats the data/removes abnormalities and returns it in a Pandas dataframe
    """
    with open(path) as f:
        lines = f.readlines()
    entries = []
    for i in range(len(lines)):
        temp_string = re.sub(r',(\d)\t', r'\t\1', lines[i])         # Fix "," issue found in "installs.txt"
        entries.append(temp_string.replace("\n",""))       # Removing "\n" from the end of each row
    
    data_columns = entries[0].split('\t')      # Seperating column names by '\'
    col_values = [[] for _ in data_columns]      # Creating lists to store column entries
    data_dict = {}
    for i in range(1,len(entries)):
        split = entries[i].split('\t')             # Splitting values by '\'
        for j in range(len(col_values)):
            col_values[j].append(split[j])
    for column_index in range(len(data_columns)):         # Creating dictionary to store column values by column names
        data_dict[data_columns[column_index]] = col_values[column_index]
    df = pd.DataFrame(data_dict)    
    return df

def normalizing_date(date_series:pd.core.series.Series,column_name:str):
    """
    This method updates the date format and outputs the difference from current date (2023-8-11), in days.
    """
    current_date = datetime.today()
    date_formatted = pd.to_datetime(date_series[column_name], format='%Y-%m-%d %H:%M:%S.%f0')
    time_diff = (current_date - date_formatted).dt.total_seconds() / (24 * 3600)                         # To calculate the difference in days
    return time_diff

# Data import and preprocessing
app_start_july_df = data_preprocessing('dataset/app starts july.txt')
app_start_df = data_preprocessing('dataset/app starts.txt')
brochure_views_july_df = data_preprocessing('dataset/brochure views july.txt')
brochure_views_df = data_preprocessing('dataset/brochure views.txt')
installs_df = data_preprocessing('dataset/installs.txt')

# merging relevant data into one dataframe
brochure_views_combined = pd.concat([brochure_views_df,brochure_views_july_df]).drop_duplicates()       #combining brochure views entries
installs_df = installs_df.drop('id', axis=1)                                                            # Removing 'id' to allow merging using 'userId'
final_df = pd.merge(brochure_views_combined, installs_df, on='userId')                                  # Merging into one dataframe for predicting churn propensity
final_df['dateCreated'] = normalizing_date(final_df,'dateCreated')                                      # Formatting time in terms of difference in days.
final_df['InstallDate'] = normalizing_date(final_df,'InstallDate')
# Saving formatted dataframe into a .csv file
output_path = 'dataset/formatted_data.csv'
final_df = final_df[~final_df['view_duration'].str.contains('NULL')]                                    # dropping entries with null/NaN values
final_df.to_csv(output_path, mode = 'a', header=not os.path.exists(output_path))		                # This will append the datafile and put headers only for the first time.)