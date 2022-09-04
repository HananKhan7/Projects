import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import pickle

def load_data_and_convert_to_Json(file_path:str)-> list:
    """
    This function loads the queries and converts them to Json format
    """
    dataset = []
    with open(file_path) as r:
        data = r.readlines()
    # Convertion to JSON
    for lines in data:
        dataset.append(json.loads(lines))
    return dataset

def load_model(model_type:str):
    """
    This method is used to select the model type (either normal BERT base or BERT fine tuned) and returns the respective model with the tokenizer
    For selecting fine tuned BERT, str = 'fine_tuned_BERT'
    """
    if model_type == 'fine_tuned_BERT':
       model = pickle.load(open('fine_tuned_BERT_Netspeak_scheduler.sav', 'rb'))
       print('fine tuned BERT selected')
    else:
        model = BertForMaskedLM.from_pretrained('bert-base-uncased',    return_dict = True)
        print('Normal BERT base selected')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, device, tokenizer

class BERT(object):
    def __init__(self, dataset:list, batch_split, model, device, tokenizer):
        """
        This constructor builds the BERT model.
        It takes the dataset, defined batch_split, model type, device type and tokenizer as parameters.
        """
        if model:
            self.dataset = dataset
            self.batch_split = batch_split
            self.model = model
            self.device = device
            self.tokenizer = tokenizer
            return
    
    def batch_creation(self, batch_index: int):
        """
        This method creates batches w.r.t to the batch_split parameter of the class
        """
        if batch_index ==0:
            return print("Please enter correct batch_index")
        data_batched = []
        batch_cut =self.batch_split* batch_index
        i=batch_cut - self.batch_split 
        for line_index in range(i,len(self.dataset)):
            if i >= batch_cut:
                break
            data_batched.append(self.dataset[line_index])
            i+=1
        return data_batched

    def extracting_data(self,data:list):
        """
        This method extracts the information from line-delimited JSON
        """
        query, length, position, source, teachers_options, teachers_answer, netspeak_options, netspeak_answer, netspeak_answers_frequency= ([] for i in range(9))
        for line in data:
            # line = json.loads(lines)
            query.append(line['query'])
            length.append(line['length'])
            position.append(line['position'])
            source.append(line['source'])
            teachers_options.append(line['teachers_options'])
            teachers_answer.append(line['teachers_answer'])
            netspeak_options.append(line['netspeak_options'])
            netspeak_answer.append(line['netspeak_answer'])
            netspeak_answers_frequency.append(line['netspeak_answers_frequency'])
        return query, length, position, source, teachers_options, teachers_answer, netspeak_options, netspeak_answer, netspeak_answers_frequency

    def BERT_implementation(self, no_of_answers:int):
        """
        This method implements BERT model, It takes the dataset and for each query, returns the number of answers based
        on the parameter : no_of_answers
        """
        batch_size = round(len(self.dataset)/self.batch_split)                   # Batch_size defines the number of batches that the model will process
        errors = []
        print("", file = open("output_BERT_ft.txt","w"))
        for i in tqdm(range(1, batch_size+1), leave = True):
            data = self.batch_creation(i)         # Method to create batches per iteration
            query, length, position, source, teachers_options, teachers_answer, netspeak_options, netspeak_answer, netspeak_answers_frequency = self.extracting_data(data)
            token_ids = self.tokenizer.batch_encode_plus(query, padding=True, truncation=True)
            token_input_ids = torch.tensor(token_ids['input_ids']).to(self.device)
            masked_position = (token_input_ids == self.tokenizer.mask_token_id).nonzero();   # Tokenizer.mask_token_id gives the index of the masks
            masked_pos = [mask[1].item() for mask in masked_position ];       # To get mask index, if needed
            with torch.no_grad():
                """
                Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward().
                It will reduce memory consumption for computations that would otherwise have requires_grad=True.
                """
                output = self.model(token_input_ids)
            last_hidden_state = output[0].squeeze()
            mask_hidden_state, idx = ([] for i in range(2))
            for query_index in range(len(masked_pos)):
                teachers_options_str = ','.join(["\""+i+"\"" for i in teachers_options[query_index]])       # Correcting the format
                i = 0
                for netspeak_response in  netspeak_answer[query_index]:
                    if netspeak_response == '\"':               #  Replacing " with QUOTATION_MARK string because of " not being detected by Json.loads()
                        netspeak_answer[query_index][i] = 'QUOTATION_MARK' 
                    i+=1
                if netspeak_options[query_index] != [[]]:                                               # Correcting the format
                    netspeak_options_str = ','.join(["\""+i+"\"" for i in netspeak_options[query_index]])
                if netspeak_answers_frequency[query_index] != [[]]:                                     # Correcting the format
                    netspeak_answers_frequency_str = ','.join(["\""+i+"\"" for i in netspeak_answers_frequency[query_index]])
                try:
                    mask_hidden_state.append(last_hidden_state[query_index][masked_pos[query_index]])
                    idx.append(torch.topk(last_hidden_state[query_index][masked_pos[query_index]], k=no_of_answers, dim=0)[1])    # Using torch.topk for getting the top  picks from the model
                    words_per_query = [self.tokenizer.decode(i.item()).strip() for i in idx[query_index]]
                    for i in range(len(words_per_query)):                 # To remove spaces between model results
                        words_per_query[i] = words_per_query[i].replace(" ", "")
                    j=0
                    for BERT_response in  words_per_query:
                        if BERT_response == '\"':               #  Replacing " with QUOTATION_MARK string because of " not being detected by Json.loads()
                            words_per_query[j] = 'QUOTATION_MARK'
                        j+=1 
                    answer_str = ','.join(["\""+i+"\"" for i in words_per_query])                       # Correcting the format
                    row = "{"+"\"query\":\"{}\",\"length\":\"{}\",\"position\":\"{}\",\"source\":\"{}\",\"teachers_options\":[{}],\"teachers_answer\":\"{}\",\"netspeak_options\":[{}],\"netspeak_answer\":\"{}\",\"netspeak_answers_frequency\":[{}],\"BERT_Results\":[{}]".format(query[query_index], length[query_index], position[query_index], source[query_index], teachers_options_str, teachers_answer[query_index], netspeak_options_str, netspeak_answer[query_index], netspeak_answers_frequency_str, answer_str)+"}"        # Single rows consisting of query along with model results and their respective scores
                    print(row, file = open("output_BERT_ft.txt","a"))
                except Exception as e:
                    row = "{"+"\"query\":\"{}\",\"length\":\"{}\",\"position\":\"{}\",\"source\":\"{}\",\"teachers_options\":[{}],\"teachers_answer\":\"{}\",\"netspeak_options\":[{}],\"netspeak_answer\":\"{}\",\"netspeak_answers_frequency\":[{}],\"BERT_Results\":[{}]".format(query[query_index], length[query_index], position[query_index], source[query_index], teachers_options_str, teachers_answer[query_index], netspeak_options_str, netspeak_answer[query_index], netspeak_answers_frequency_str, [])+"}"        # Single rows consisting of query along with model results and their respective scores
                    errors.append((query, str(e))) 
        print("Errors: ","\n", errors, file = open("output_BERT_ft.txt", "a")) # Saving the query with the error along with the string of error at the end of the .txt file


if __name__ == "__main__":
    dataset = load_data_and_convert_to_Json('single-queries.txt')
    model, device, tokenizer = load_model('fine_tuned_BERT')
    # Defining the batch_split
    batch_split = 500
    # Creating BERT object
    BT = BERT(dataset, batch_split, model, device, tokenizer)
    BT.BERT_implementation(no_of_answers=30)

