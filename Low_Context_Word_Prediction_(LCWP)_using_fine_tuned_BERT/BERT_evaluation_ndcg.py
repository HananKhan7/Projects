import json
import torch
from tqdm import tqdm
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt 
import math
from matplotlib.pyplot import figure

def load_data_and_convert_to_Json(file_path:str)-> list:
    """
    This function loads the queries and converts them to Json format
    """
    dataset = []
    errors = []
    try:
        with open(file_path, encoding="utf8") as r:          
            data = r.readlines()
        # Convertion to JSON
        for lines in data:
            dataset.append(json.loads(lines))
    except Exception as e:
        errors.append(str(e))
    return dataset

def select_device():
    """
    This method selects cuda device to implement the processing on gpu for more efficieny w.r.t speed.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

class Evaluation(object):
    def __init__(self, device, evaluation_type:str, dataset_ft:list, dataset_normal:list):
        """
        This constructs the object for the Evaluation class.
        parameter evaluation_type: this determines which type of evaluation to conduct out of the three options
            1- percentage 
            2- threshold
            3- logarithmic
        parameter dataset_ft & dataset_normal: these are the datasets generated from fine-tuned BERT and BERT normal case.
        """
        self.device = device
        self.evaluation_type = evaluation_type
        self.dataset_ft = dataset_ft
        self.dataset_normal = dataset_normal
        self.preprocessing_data()                   #This automatically filters the dataset based on the method 'filtering_dataset'
        return

    
    def percentage(self,netspeak_answers:list, netspeak_answers_freq: list):
        """
        This method uses the percentage gotten from number of occurances of BERT results and then based on the percentages,
        creates a percentage threshold.
        """
        freq_sum = sum(netspeak_answers_freq)
        percents = [num/freq_sum for num in netspeak_answers_freq]
        scores = [4 if percent >= 0.1 else
                3 if 0.1 > percent >= 0.05 else
                2 if 0.05 > percent >= 0.02 else
                1 for percent in percents]
        return dict(zip(netspeak_answers, scores))


    def threshold(self,netspeak_answers:list, netspeak_answers_freq: list):
        """
        This method uses hardcoded threshold values for defining answers.
        """
        scores = [3 if freq >= 1e5 else
                2 if 1e5 > freq >= 1e4 else
                1 for freq in netspeak_answers_freq]
        return dict(zip(netspeak_answers, scores))


    def logarithmic(self, netspeak_answers:list, netspeak_answers_freq: list):
        """
        This method uses log to power of 5 to create more thresholds.
        """
        scores = [math.ceil(math.log(freq,5)) for freq in netspeak_answers_freq]
        return dict(zip(netspeak_answers, scores))
    
    def ndcg(self, model_answers:list, netspeak_answers:list, netspeak_answers_freq: list):
        """
        This method calculates the NDCG based on the evaluatiuon type selected.
        """
        if self.evaluation_type == 'percentage':
            netspeak_scores = self.percentage(netspeak_answers, netspeak_answers_freq)
        elif self.evaluation_type == 'threshold':  
            netspeak_scores = self.threshold(netspeak_answers, netspeak_answers_freq)
        elif self.evaluation_type == 'logarithmic':  
            netspeak_scores = self.logarithmic(netspeak_answers, netspeak_answers_freq)
        else:
            print('Please select correct evaluation type')
            return
        dcg = [netspeak_scores[answer] if answer in netspeak_scores.keys() else 1 for answer in model_answers]
        idcg = list(netspeak_scores.values())
        idcg = idcg + [1]*(len(dcg)-len(idcg))
        dcg = sum([y/math.log2(x+2) for x,y in enumerate(dcg)])
        idcg = sum([y/math.log2(x+2) for x,y in enumerate(idcg)])
        return dcg/idcg

    def filtering_dataset(self,dataset:list):
        """
        This method converts queries to JSON format and only selects the one with netspeak answers
        """
        data = []
        errors = []
        for query in dataset:
            try:
                answer = query['netspeak_answer']
                if answer != '[]':
                    data.append(query)
            except Exception as e:
                errors.append([query, str(e)])
        return data
    
    def preprocessing_data(self):
        """
        This method applies the filtering process to the respective datasets.
        """
        self.dataset_ft = self.filtering_dataset(self.dataset_ft)
        self.dataset_normal = self.filtering_dataset(self.dataset_normal) 

    def calculating_ndcg(self,data:list):
        """
        This method calculates the ndcg ranking based on the results attained from the respective model. 
        """
        count = [[0]*10 for i in range(10)]
        ndcg_total = [[0]*10 for i in range(10)]
        ndcg_BERT = []
        errors = []
        for query in tqdm(data,leave = True):
            try:
                options = query['netspeak_options']
                frequency = query['netspeak_answers_frequency']
                frequency = [ int(x) for x in frequency ]
                length = int(query['length'])
                position = int(query['position'])
                result = query['BERT_Results']
                ndcg_result = self.ndcg(result, options, frequency)
                ndcg_total[length][position] += ndcg_result
                count[length][position] += 1 
            except Exception as e:
                errors.append(str(e))
        sum_ndcg = 0
        sum_count = 0
        for i in range(3,10):
            for j in range(i):
                sum_ndcg += ndcg_total[i][j]
                sum_count += count[i][j]
                if count[i][j] > 0:
                    ndcg_BERT.append([i,j,ndcg_total[i][j] / count[i][j]])         # Length of query, position of mask, nDCG value
                else:
                    ndcg_BERT.append([i,j,"-"])
        ndcg_BERT.append(sum_ndcg/sum_count)
        return ndcg_BERT

    def get_ndcg_scores(self):
        """
        This method implements 'calculating_ndcg' on datasets obtained from fine-tuned BERT and normal BERT.
        """
        ndcg_ft = self.calculating_ndcg(self.dataset_ft)
        ndcg_normal = self.calculating_ndcg(self.dataset_normal)
        return ndcg_ft, ndcg_normal
    
    def calculating_scores(self, ndcg:list):
        """
        This method extracts three, four and five grams NDCG scores out of all the scores because
        evalution will take place w.r.t Netspeak answers and netspeak does not provide answers
        for queries with length > 5.
        """
        three_gram_scores = []
        four_gram_scores = []
        five_gram_scores = []
        for i in range(len(ndcg)-1):
            three_gram_scores.append(ndcg[i][2])  if ndcg[i][0] == 3 else four_gram_scores.append(ndcg[i][2]) \
                if ndcg[i][0] == 4 else five_gram_scores.append(ndcg[i][2]) if ndcg[i][0] == 5 else []
        return three_gram_scores, four_gram_scores, five_gram_scores
    
    def plot(self):
        """
        This method is used to create a bar chart to compare the ndcg gotten from fine-tuned BERT and normal BERT
        """
        ndcg_ft, ndcg_normal = self.get_ndcg_scores()
        three_gram_scores_ft, four_gram_scores_ft, five_gram_scores_ft = self.calculating_scores(ndcg_ft)
        three_gram_scores_normal, four_gram_scores_normal, five_gram_scores_normal = self.calculating_scores(ndcg_normal)
        figure(figsize=(10, 20), dpi=80)
        evaluation_str = ','.join(["\""+self.evaluation_type+" type evaluation\""]) 
        plt.suptitle(evaluation_str, fontsize=25)
        # For 3 grams
        plt.subplot(3,1,1)
        x_axis = np.arange(len(three_gram_scores_normal))
        plt.bar(x_axis -0.2, three_gram_scores_normal, width=0.4, label = 'BERT-base')
        plt.bar(x_axis +0.2, three_gram_scores_ft, width=0.4, label = 'BERT-fine-tuned')
        axs = plt.gca()
        axs.set_ylim([min(three_gram_scores_normal)-0.25, max(three_gram_scores_ft)+0.25])
        axs.set_title("nDCG 3-grams")
        axs.set_xlabel("MASK position")
        axs.set_ylabel("score")
        plt.legend()

        # For 4 grams
        plt.subplot(3,1,2)
        x_axis = np.arange(len(four_gram_scores_normal))
        plt.bar(x_axis -0.2, four_gram_scores_normal, width=0.4, label = 'BERT-base')
        plt.bar(x_axis +0.2, four_gram_scores_ft, width=0.4, label = 'BERT-fine-tuned')
        axs = plt.gca()
        axs.set_ylim([min(four_gram_scores_normal)-0.25, max(four_gram_scores_ft)+0.25])
        axs.set_title("nDCG 4-grams")
        axs.set_xlabel("MASK position")
        axs.set_ylabel("score")
        plt.legend()

        # For 5 grams
        plt.subplot(3,1,3)
        x_axis = np.arange(len(five_gram_scores_normal))
        plt.bar(x_axis -0.2, five_gram_scores_normal, width=0.4, label = 'BERT-base')
        plt.bar(x_axis +0.2, five_gram_scores_ft, width=0.4, label = 'BERT-fine-tuned')
        axs = plt.gca()
        axs.set_ylim([min(five_gram_scores_normal)-0.25, max(five_gram_scores_ft)+0.25])
        axs.set_title("nDCG 5-grams")
        axs.set_xlabel("MASK position")
        axs.set_ylabel("score")
        plt.legend()
        plt.show
        evaluation_str = ','.join([self.evaluation_type+"_type_evaluation.jpg"])
        plt.savefig(evaluation_str)

if __name__ == "__main__":
    dataset_ft = load_data_and_convert_to_Json('output_BERT_ft.txt')
    dataset_normal = load_data_and_convert_to_Json('output_BERT.txt')
    ev = Evaluation(device = select_device(), evaluation_type='percentage', dataset_ft= dataset_ft, dataset_normal= dataset_normal)
    ev.plot()
    
    