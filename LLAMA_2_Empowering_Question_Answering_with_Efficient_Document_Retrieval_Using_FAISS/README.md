# LLM_Task

This repository contains a powerful Question Answering (QA) system built using the Large Language Model, LLAMA 2. The system is designed to extract accurate and relevant answers from a set of provided local documents. It utilizes FAISS for efficient word embeddings, enabling fast and accurate retrieval of information.

## Prerequisites
- python 3.x
- The required libraries can be found in the requirement.txt file

## Implementation
The project is available in both a python script and a jupyter notebook file.

It consists of the following main tasks.
### Data preprocessing
- The provided article is saved into a .txt format in 'raw_data' folder which is then cleaned up by undergoing the following processes:
    - Lowercasing
    - Special characters/HTML links removal
    - removal of new lines
- It is then printed into a seperate .txt file placed in the 'preprocessed_data' folder.

### Word embeddings
Word embeddings are generated for the preprocessed data using one of the hugging face's most popular sentence transformers **all-mpnet-base-v2** the embeddings are stored in a local database using FAISS.

### LLM LLAMA 2
One of the variants of LLAMA 2 named as **llama-2-7b-chat.ggmlv3.q8_0] is used for building a question answer system oriented towards local data.
Two other variants of LLAMA 2 were also tested named **llama-2-7b-chat.ggmlv3.q3_K_M** and **llama-2-7b-chat.ggmlv3.q6_K**. But the above mentioned model was selected based on its performance.

#### Important note!
Due to the large size of models, I was not able to upload them on Git. Due to which, the folder "Models" is empty. However these models can be accessed/downloaded through [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main). Make sure to place the model file into the "Models" folder (In GGML format).

### Prompting LLM model
Finally the LLM model was prompted with user queries/questions. The output of the model consisting of users question, model's generated answer as well as the source document is printed as well as stored in a .txt file named as "output.txt" in "model_results" folder.

## References
[LLAMA 2 documentation](https://ai.meta.com/llama/)

[LLAMA 2 hugging face](https://huggingface.co/meta-llama/Llama-2-7b)

[LLAMA 2 guide](https://huggingface.co/blog/stackllama#stackllama-a-hands-on-guide-to-train-llama-with-rlhf)

[LLAMA 2 models](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

[C Transformers](https://python.langchain.com/docs/integrations/llms/ctransformers)

[Sentence transformers](https://huggingface.co/sentence-transformers)

[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

[QA using a retreiver](https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa)

[Langchain](https://www.langchain.com/)

[Langchain Documents](https://python.langchain.com/docs/modules/chains/document/)

[FAISS semantic search](https://huggingface.co/learn/nlp-course/chapter5/6?fw=pt)