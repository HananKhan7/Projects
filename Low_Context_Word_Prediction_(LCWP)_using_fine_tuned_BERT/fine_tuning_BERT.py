import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup, AdamW
from torch.nn import functional as F
import re
import random
import pickle
from sklearn.model_selection import train_test_split


def load_data_and_convert_to_Json(file_path:str)-> list:
    """
    This function loads the queries and converts them to Json format
    """
    dataset = []
    errors = []
    with open(file_path) as r:
        data = r.readlines()
    # Convertion to JSON
    for lines in data:
        try:
            dataset.append(json.loads(lines))
        except Exception as e:
            errors.append([lines, str(e)])
    return dataset

def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict = True)
    model.to(device)
    tokenizer =BertTokenizer.from_pretrained('bert-base-uncased')
    return model, device, tokenizer

class Fine_Tuning_BERT(object):
    def __init__(self, dataset:list, model, device, tokenizer):
        """
        This constructor builds the fine-tuning-model.
        It takes the dataset which is converted into JSON format, as parameters.
        """
        if model:
            self.dataset = dataset
            self.model = model
            self.device = device
            self.tokenizer = tokenizer
            return

    def filtering_queries(self) ->list:
        """
        This method is used for selecting only the queries which have netspeak answers, it ignores the queries
        with empty netspeak answers
        """
        data = []
        errors = []
        for queries in self.dataset:
            try:
                answer = queries['netspeak_answer']
                if answer != '[]':
                    data.append(queries)
            except Exception as e:
                errors.append([queries, str(e)])
        return data

    def filling_queries(self):
        """
        This method is used to fill up the [MASK]'s with the results from Netspeak
        """
        original_mask = '[MASK]'
        errors = []
        for files in tqdm(self.filtering_queries() , leave = True):
            try:
                query = files['query']                    # article in the dataset
                options = files['netspeak_options']
                answer = files['netspeak_answer']
                mask_count = 0
                result = re.findall(original_mask,query)                                   # Find all the possible [MASK] positions                                           
                query = query.replace(original_mask,answer,1)                    # Replacing original mask with correct teacher answers
                mask_count +=1
                files['query'] = query
            except Exception as e:
                errors.append([files, str(e)])
    
    def processing_data(self, N:int) -> list:
        """
        This method selects the N number of queries from the total dataset to train the model.
        Parameter N: number of queries to train the model on.
        """
        self.filling_queries()
        data = random.choices(self.dataset,k=N)
        # Combining the processed queries
        queries = []
        for i in range(len(data)):
            queries.append(data[i]['query'])
        return queries

    def preparing_inputs(self):
        """
        This method generates the input BatchEncoding to be used as a parameter for pytorch dataset object
        """
        queries = self.processing_data(50000)
        inputs = self.tokenizer(queries, return_tensors='pt', max_length=512, truncation=True, padding='max_length') # This will either truncate or pade each article to 512 length
        inputs['labels'] = inputs.input_ids.detach().clone()    # Labels are created to store the original articles without the masks
        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        # create mask array,  setting 30 % threshold for masking (values under 0.03 values apart from the special tokens will be masked)
        # Rules to ignore the special tokens (classifiers, seperators and padding tokens)
        mask_arr = (rand < 0.30) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)              
        # Saving all of the masked token's indexes
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
        #Then apply these indices to each respective row in input_ids, assigning each of the values at these indices as 103.
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        return inputs
    
    def test_train_splitting(self, test_data_size:int, test_batch_size:int, train_batch_size:int):
        """
        This method splits the dataset into training and testing dataset based on the split defined by
        the parameter test_data_split.
        parameter test_batch_size: This defines the number of batches for testing dataset
        parameter train_batch_size: This defines the number of batches for training dataset
        """
        inputs = self.preparing_inputs()
        dataset = MeditationsDataset(inputs)
        # Splitting into training and testing
        X_train, X_test = train_test_split(dataset, test_size=test_data_size, shuffle = True, random_state=42)          # Test_size = 0.3
        training_dataset = torch.utils.data.DataLoader(X_train, batch_size=test_batch_size, shuffle=True)          # Shuffling the selection of batches 
        testing_dataset = torch.utils.data.DataLoader(X_test, batch_size=train_batch_size, shuffle=True)          # Shuffling the selection of batches
        return training_dataset, testing_dataset
    
    def fine_tuning_implementation(self,test_data_size:int, test_batch_size:int, train_batch_size:int, lr: int, adam_epsilon:int, num_warmup_steps:int, epochs:int):
        """
        This method fine tunes BERT on the given dataset,based on the hyperparameters, and saves the fine-tuned-model at the end.
        parameter adam_epsilon: this is a very small number to prevent any division by zero in the implementation
        parameter num_warmup_steps: creates a schedule with a learning rate that decreases linearly after 
        linearly increasing during a warm-up period. (so that it decreases when we get close to finding minimum)
        parameter epochs: defines the number of times the dataset laps through the training phase.
        """
        training_dataset, testing_dataset = self.test_train_splitting(test_data_size, test_batch_size, train_batch_size)
        # activate training mode
        self.model.train()
        # initialize optimizer
        num_training_steps = len(training_dataset)*epochs
        optim = AdamW(self.model.parameters(), lr=lr,eps = adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
        # Initial Loss
        print('Calculating the loss for standard base model')        

        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(testing_dataset, leave=True)
            for batch in loop:          # For each batch in our dataset
                # initialize calculated gradients (from prev step)      To start with zero gradient
                optim.zero_grad()
                # pull all tensor batches required for training, token_type_ids are not required for masked modeling
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                labels=labels)
                # extract loss
                loss = outputs.loss
                # print relevant info to progress bar
                loop.set_description(f'validation dataset Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        print('Fine-tuning the model')       

        for epoch in range(epochs):
            # setup loop with TQDM and dataloader
            loop = tqdm(training_dataset, leave=True)
            for batch in loop:          # For each batch in our dataset (set to 16)
                # initialize calculated gradients (from prev step)      To start with zero gradient
                optim.zero_grad()
                # pull all tensor batches required for training, token_type_ids are not required for masked modeling
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                # process
                outputs = self.model(input_ids, attention_mask=attention_mask,
                                labels=labels)
                # extract loss
                loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # Update learning rate schedule
                scheduler.step()
                # print relevant info to progress bar
                loop.set_description(f'Training dataset Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        # Saving the trained model
        filename = 'fine_tuned_BERT.sav'
        pickle.dump(self.model, open(filename, 'wb'))


class MeditationsDataset(torch.utils.data.Dataset):
    """
    This class formats the BatchEncodings for it to be compatible with the DataLoader from PyTorch.
    """
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):     # So that the dataloader can get the dictionary formated batch of these items
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}         # to return the input_ids, token_types, attantion_mask and label's keys  
    def __len__(self):              # So that the dataloader can get the length of the dataset its looking at
        return len(self.encodings.input_ids)

if __name__ == "__main__":
    dataset = load_data_and_convert_to_Json('output-Netspeak.txt')
    model, device, tokenizer = load_model()
    ft = Fine_Tuning_BERT(dataset, model, device, tokenizer)
    ft.fine_tuning_implementation(test_data_size=0.3,test_batch_size=1, train_batch_size=8, lr=1e-5, adam_epsilon=1e-8, num_warmup_steps=2, epochs=4)



