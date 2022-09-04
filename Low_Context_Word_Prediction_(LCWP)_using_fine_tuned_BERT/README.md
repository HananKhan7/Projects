# Project Low-Context Word Predictions WS2021/22

## Introduction
The goal of this project was to use BERT (Fine-tuned) in order to implement low context word prediction on a set of cloze test dataset which was pre processed. 
## Pre-processing

### Original Dataset
The dataset used for this project is a large-scale human created cloze test dataset named CLOTH which is collected from English exams. Questions in the dataset are designed by middle-school and high-school teachers to prepare Chinese students for entrance exams with missing blanks carefully created by teachers and candidate choices purposely designed to be nuanced. The dataset can be found at <https://github.com/qizhex/Large-scale-Cloze-Test-Dataset-Created-by-Teachers>

### Prerpocessing
The data in its original form was not suitable to be used for analysis with BERT model and therefore, it needed to be pre-processed. The fill in the blanks 'underscores' were replaced by '[MASK]' so that BERT model would be able to identify the masks. The data was lowercased, additional spaces were removed along with punctuations. 
### Obtaining Netspeak output
The output obtained from the preprocessing stage resulted in a *single-queries.txt* file which contains all the queries with a single MASK from both middle and high school categories. These queries were used as an input to obtain, when possible, 30 number of answers along with their number of occurences from Netspeak by means of *Netspeak_Script.py* python script. The output was saved into a text file in a line delimited JSON format as previously done during the preprocessing phase. In order to maintain an efficient Git repository, only 10k output queries are shown. Output for a single query from Netspeak can be seen below.
```python
{"query":"My heart [MASK]",  
"length":"3",  
"position":"2",  
"source":"high0",  
"teachers_options":["ached","beat","sank","rose"],  
"teachers_answer":"sank",  
"netspeak_options": [".","is",",","and","was","to","i","that","will","in","for",";","!","goes","with","beat","has","sank","out","as","would","?","lyrics","of","the","at","on","belongs","beats","you"],   "netspeak_answer": ".",  
"netspeak_answers_frequency":["1512194","846610","688435","672548","454055","448625","251084","219633","205496","174502","160410","159738","149477","146513","143640","102625","100204","88209","79264","78106","76542","72819","72204","64346","62434","62420","61099","54787","53909","52447"]}  
```

## BERT_fine_tuned

### Introduction
BERT(Bidirectional Encoder Representations from Transfromers) is, as the name mentions, a bidirectional model which is primarily trained for masked language prediction and next sentence prediction. Due to this, it suited as a good candidate for this project. BERT is initially trained on high context data and since the goal of the project was to achieve low context word prediction, fine-tuning was required.

### Fine-tuning BERT
As previously mentioned, Since BERT is initially trained on high context, fine-tuning was necessary in order to meet the requirements of the project. BERT was fine tuned on small portion of the dataset (50k queries) by using *fine_tuning_BERT,py* script. Step by step details can be found with in the script itself. Hyperparameters such as batch size for testing and training data, learning rate, adam_epsilon, number of warmup steps and number of epochs were decided by conducting multiple trials on different values and selecting the ones with the best results. Finally after fine-tuning BERT, the model was saved and stored in the desired directory by using *pickle* library which can be simply installed in the following way.

```python
pip install pickle
import pickle
```
### Implementing BERT (In Batches)
After fine-tuning BERT, queries were passed through both, normal BERT base (uncased) and fine-tuned BERT, in batches (500 queries per batch) to get outputs from both models to compare the performance of fine-tuned-BERT against normal BERT-base-uncased. The reason for batching was to improve the pace of the model, Instead of passing a single query through BERT in each iteration, 500 queries were passed. Which resulted in a substantial reduction of processing time. Implementation of the model was done by *Batching_BERT.py* script. Step by step details can be found with in the script. The output from the models was stored again, into a text file in a line delimited JSON format. Results of only 10k queries are shown in Git to keep the repository efficient. Output from a single query from both normal BERT-base-uncased and fine-tuned BERT can be seen below.
#### BERT-base-uncased
```python
{"query":"My heart [MASK]",
"length":"3",
"position":"2",
"source":"high0",
"teachers_options":["ached","beat","sank","rose"],
"teachers_answer":"sank",
"netspeak_options":[".","is",",","and","was","to","i","that","will","in","for",";","!","goes","with","beat","has","sank","out","as","would","?","lyrics","of","the","at","on","belongs","beats","you"],"netspeak_answer":".",
"netspeak_answers_frequency":["1512194","846610","688435","672548","454055","448625","251084","219633","205496","174502","160410","159738","149477","146513","143640","102625","100204","88209","79264","78106","76542","72819","72204","64346","62434","62420","61099","54787","53909","52447"],
"BERT_Results":[".","|",";","!","?","।","...","॥","[UNK]","-",":",")","~","}","。","is","##¤","'","QUOTATION_MARK",",","beat","defaulted","．","##¦","##hita","=","(","##¨","¤","#"]}
```
#### Fine-tuned-BERT
```python
{"query":"My heart [MASK]",
"length":"3",
"position":"2",
"source":"high0",
"teachers_options":["ached","beat","sank","rose"],
"teachers_answer":"sank",
"netspeak_options":[".","is",",","and","was","to","i","that","will","in","for",";","!","goes","with","beat","has","sank","out","as","would","?","lyrics","of","the","at","on","belongs","beats","you"],"netspeak_answer":".",
"netspeak_answers_frequency":["1512194","846610","688435","672548","454055","448625","251084","219633","205496","174502","160410","159738","149477","146513","143640","102625","100204","88209","79264","78106","76542","72819","72204","64346","62434","62420","61099","54787","53909","52447"],
"BERT_Results":["is","and","was",".",",","beat","to","in","stopped","but","beating","of","rate","still","##felt","so","that","not","!","beats","as","broke","has","he","just","i","had","at","pounding","with"]}
```
### Evaluation process
NDCG (Normalized Discounted Cumulative Gain) was used for the evaluation process. Furthermore, three techniques were implemented to categorize the answers for this evaluation process.
#### Percentage approach
This method looks into the number of occurances of the results gotten from BERT and based on those occurances, creates a percentage threshold.
#### Threshold approach
This method uses hardcoded threshold values for sepreating answers from options and other categories.
#### Logarithimic approach
This method uses log to the power of 5 to genereate more than just 3 categories for answers.

More details about the evaluation process can be found in *BERT_evaluation_ndcg.py* script.

### Result
NDCG results obtained from the three approaches, for both normal and fine-tuned BERT, can be seen below in the form of bar charts.
#### Percentage approach
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Low_Context_Word_Prediction_(LCWP)_using_fine_tuned_BERT/Outputs/percentage_type_evaluation.jpg)
#### Threshold approach
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Low_Context_Word_Prediction_(LCWP)_using_fine_tuned_BERT/Outputs/threshold_type_evaluation.jpg)
#### Logarithimic approach
![ScreenShot](https://github.com/HananKhan7/Projects/blob/main/Low_Context_Word_Prediction_(LCWP)_using_fine_tuned_BERT/Outputs/logarithmic_type_evaluation.jpg)

