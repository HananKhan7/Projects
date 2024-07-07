"""
Project: Personalized Toxicity Detection Using Mixtral-8x7B-Instruct-v0.1 and User Profiles

Description:
This project aims to enhance the accuracy and personalization of toxicity classification by utilizing the Mixtral-8x7B-Instruct-v0.1 model.
The dataset comprises toxic comments sourced from multiple social media platforms. Each comment is annotated by five annotators, each providing a toxicity score.
Additionally, each annotator has 17 user profile traits that define their personality.

The core objective is to incorporate these personality traits into the model through prompt engineering, thereby producing more personalized and accurate toxicity classifications.
For enhanced contextual understanding, the user profiles and annotations of four other annotators for the same comments are also provided to the model.
This context allows the model to better grasp the nuances and perspectives of different annotators, resulting in more refined toxicity assessments.

Key Features:
- Utilizes Mixtral 7B model for toxicity classification.
- Incorporates 17 personality traits of annotators into the model.
- Provides contextual information by including user profiles and annotations of four other annotators for the same comments.
- Aims to produce personalized toxicity scores based on individual annotator profiles.

Usage:
- Processes the dataset.
- Integrates the personality traits via prompt engineering.
- Feeds the contextual information into the Mixtral 7B model.
- Generates personalized toxicity classifications.

Author: Abdul Hanan Khan
"""




# Importing the libraries
import os
import json
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Parameters
access_token = '****'
cache_dir = "Mixtral_7B_quantized/DP-checkpoints"
N_shots = 5 # Number of annotatated comments to include in the training data via prompt.
max_annotators = 2000
output_path = 'outputs/personalization_by_profile/output_files_quantized/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# Hugging Face model checkpoint
checkpoint = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load model and tokenizer from Hugging Face
quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map="auto",
    quantization_config=quantization_config,
    cache_dir=cache_dir,
    use_auth_token=access_token
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=access_token, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load the dataset
with open("dataset/annotator_datasets.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

# Function to predict toxicity using the model
def predict_toxicity(comment, primary_profile, secondary_profiles, primary_train_data, secondary_train_data_list):
    """
    This function uses the Hugging Face model to predict the toxicity rating of a given comment.
    It constructs a prompt with the primary annotator's profile, example comments, and secondary annotators' profiles.
    The function then processes the model output and returns the predicted toxicity rating.

    Parameters:
    comment (str): The comment for which the toxicity rating needs to be predicted.
    primary_profile (dict): The profile of the primary annotator.
    secondary_profiles (list): A list of profiles of secondary annotators.
    primary_train_data (list): A list of example comments annotated by the primary annotator.
    secondary_train_data_list (list): A list of lists, where each inner list contains example comments annotated by a secondary annotator.

    Returns:
    str: The predicted toxicity rating of the comment.
    """
    # Construct primary profile details and example texts
    example_texts_primary = "\n".join([f'Text: "{entry["comment"]}"\nRating: {entry["toxic_score"]}' for entry in primary_train_data])

    # Construct secondary examples texts
    additional_examples_texts = ""
    annotator_labels = ["B", "C", "D", "E"]

    for idx, (secondary_profile, secondary_train_data) in enumerate(zip(secondary_profiles, secondary_train_data_list)):
        additional_examples_text = "\n".join([f'- Text: "{entry["comment"]}"\n    Rating: {entry["toxic_score"]}' for entry in secondary_train_data])
        
        additional_examples_texts += f"""[INST]For context, here are the same comments rated by a {annotator_labels[idx]} person with a different profile:\n Their profile: gender: '{secondary_profile['gender']}', 
        race: '{secondary_profile['race']}', age range: '{secondary_profile['age_range']}', 
        education level: '{secondary_profile['education']}', political affiliation: '{secondary_profile['political_affilation']}', 
        LGBTQ status: '{secondary_profile['lgbtq_status']}', parental status: '{secondary_profile['is_parent']}', 
        importance of religion: '{secondary_profile['religion_important']}', impact of technology: '{secondary_profile['technology_impact']}', 
        {'They use social media' if secondary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if secondary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if secondary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if secondary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if secondary_profile['uses_media_forums'] else 'They do not identify as transgender'},
        - They find toxic comments to be {secondary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if secondary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if secondary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        \nexamples: {additional_examples_text}[/INST]\n"""

    prompt = f"""
        <s>
        [INST]
        Given a text, how would a person 'A' of gender: '{primary_profile['gender']}', 
        race: '{primary_profile['race']}', age range: '{primary_profile['age_range']}', 
        education level: '{primary_profile['education']}', political affiliation: '{primary_profile['political_affilation']}', 
        LGBTQ status: '{primary_profile['lgbtq_status']}', parental status: '{primary_profile['is_parent']}', 
        importance of religion: '{primary_profile['religion_important']}', impact of technology: '{primary_profile['technology_impact']}', 
        {'They use social media' if primary_profile['uses_media_social'] else 'They do not use social media'}, 
        {'They use news from media' if primary_profile['uses_media_news'] else 'They do not use news from media'}, 
        {'They use video from media' if primary_profile['uses_media_video'] else 'They do not use video from media'}, 
        {'They use forums' if primary_profile['uses_media_forums'] else 'They do not use forums'}, 
        and {'They identify as transgender' if primary_profile['uses_media_forums'] else 'They do not identify as transgender'}, 
        rate the degree of toxicity in the given text, considering:
        - They find toxic comments to be {primary_profile['toxic_comments_problem']}
        - { 'They have personally seen toxic content' if primary_profile['personally_seen_toxic_content'] else 'They have personally not seen toxic content' }
        - { 'They have personally been a target of toxicity' if primary_profile['personally_been_target'] else 'They have personally not been a target of toxicity' }.
        The predefined categories are: 0 : not toxic, 1 : slightly toxic, 2 : moderately toxic, 3 : very toxic, 4 : extremely toxic.
        Only respond with the single numerical rating without including the word "Toxicity". Do not provide explanations, empty strings, or notes.
        Here are some text examples, rated by person 'A': 
        {example_texts_primary} [/INST]
        {additional_examples_texts}
        [INST]Now classify the following text based on person 'A'.
        Text: {comment} [INST]</s>
        [INST] Toxicity:
        """

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length = len(prompt))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            top_p=1, top_k=1, temperature=1
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract the numerical rating from the output
    numerical_rating = decoded_output.split()[-1]
    return numerical_rating

# Function to create annotator output
def create_annotator_output(dataset, max_annotators, N_shots, output_path):
    """
    This function processes a dataset of comments and ratings, and generates a JSON file containing the model's predictions and annotator ratings.

    Parameters:
    dataset (list): A list of dictionaries, where each dictionary represents an annotator and their datasets. Each annotator dictionary should have the following keys: 'annotator_id', 'annotator_profile', 'test_dataset', and 'train_dataset'.
    output_path (str): The path where the generated JSON file will be saved.
    max_annotators (int): The maximum number of unique annotators to consider.
    N_shots (int): The number of shots to use as training data in prompt, for each annotator.

    Returns:
    None
    """

    # Initialize an empty dictionary to store the annotator data
    annotator_data = {}

    # Initialize the tqdm progress bar for annotators
    annotator_progress = tqdm(total=max_annotators, desc="Processing annotators")

    # Iterate over each annotator in the dataset
    for i, annotator in enumerate(dataset):
        if i >= max_annotators:
            break

        annotator_id = annotator['annotator_id']
        primary_profile = annotator['annotator_profile'][0]
        test_dataset = annotator['test_dataset']
        primary_train_dataset = annotator['train_dataset'][:N_shots]  # Select N_shots examples

        # Find secondary annotators who have annotated all N_shots comments
        secondary_profiles = []
        secondary_train_data_list = []
        primary_train_comment_ids = {entry['comment_id'] for entry in primary_train_dataset}
        for secondary_annotator in dataset:
            if secondary_annotator['annotator_id'] != annotator_id and len(secondary_profiles) < 4:
                secondary_train_comments = {entry['comment_id']: entry for entry in secondary_annotator['train_dataset']}
                secondary_test_comments = {entry['comment_id']: entry for entry in secondary_annotator['test_dataset']}
                combined_secondary_comments = {secondary_train_comments, secondary_test_comments}
                if primary_train_comment_ids.issubset(combined_secondary_comments.keys()):
                    secondary_profiles.append(secondary_annotator['annotator_profile'][0])
                    secondary_train_data_list.append([combined_secondary_comments[comment_id] for comment_id in primary_train_comment_ids])

        if len(secondary_profiles) < 4:
            print(f"Not enough secondary annotators found for annotator {annotator_id} with the required comments.")
            continue

        # Add annotator data
        annotator_data[annotator_id] = {
            "annotator_id": annotator_id,
            "annotator_profile": primary_profile,
            "ratings": []
        }

        # Iterate over each comment in the test_dataset
        for entry in test_dataset:
            comment_id = entry['comment_id']
            comment = entry['comment']
            toxic_score = entry['toxic_score']

            # Get the model's prediction for the comment
            prediction = predict_toxicity(comment, primary_profile, secondary_profiles, primary_train_dataset, secondary_train_data_list)

            # Create a dictionary to store the rating data
            rating_data = { 
                "comment_id": comment_id,
                "comment": comment,
                "model_prediction": prediction,
                "toxic_score": toxic_score
            }

            # Append the rating data to the annotator's data
            annotator_data[annotator_id]['ratings'].append(rating_data)

        annotator_progress.update(1)

    annotator_progress.close()

    # Save the collected data to a JSON file
    with open(output_path, 'w') as f:
        json.dump(list(annotator_data.values()), f, indent=4)


# Process the data and create the output file

create_annotator_output(data, max_annotators=max_annotators, N_shots=N_shots, output_path=f"{output_path}toxicity_classification_pbp_{N_shots}_shots_with_additional_profiles.json")
