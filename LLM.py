# Importing libraries
import re, os
import string
from nltk.tokenize import word_tokenize
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# Function to preprocess text
def preprocess_text(input_path, output_path):
    """
    parameters:
        - input_path: Input data (article) directory path
        - output_path: Path for printing the processed data.

    This function preprocesses the input file (provided artice) in the following manner:
        - Lowercases the data
        - Remove any special characters/HTML tags
        - removes new line seperation
    """
    with open(input_path, "r", encoding="utf-8") as file:
        text = file.read()
    # Convert to lowercase
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = " ".join(
        word_tokenize(text.translate(str.maketrans("", "", string.punctuation)))
    )  # Remove new lines
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


# For generating embeddings and storing in local database using FAISS
def database_embeddings(data_path, chunk_size, chunk_overlap, save_local):
    """
    parameters:
        - data_path: Directory consisting of preprocessed input file (can be multiple)
        - chunk_size: chunk size of text fed into the embeddings model
        - chunk_overlap: Amount of chunk overlap
        - save_local: boolean variable for storing FAISS database localy
    output:
        - database containing word embeddings

        This function utilizes hugging face embeddings model through langchain framework,
        the most popular and efficient model was selected.
        FAISS is used to store the embeddings into a local database.
    """
    # define which documents to load
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)

    # splitting into chunks based on defined variable
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = splitter.split_documents(document)
    # Alternative approach would be too use transformers library with auto tokenizer from a pretrained model (same model as below can be used)
    # Generating embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
    )

    # create and save the local database using FAISS
    db = FAISS.from_documents(texts, embeddings)
    # To store database (If needed)
    if save_local:
        db.save_local("faiss")
    return db


# Generating pre-trained LLM to answer users questions based on local content
def QA_model(db, model, max_tokens, temperature):
    """
    parameters:
        - db: Database containing word embeddings
        - model: Option to choose a large language model from different LLAMA 2 variants. (other models can also be used),
                  currently three available locally in 'Models' directory:
                    llama-2-7b-chat.ggmlv3.q8_0 (latest and best performance) , llama-2-7b-chat.ggmlv3.q6_K       llama-2-7b-chat.ggmlv3.q3_K_M
        - max_tokens: Max number of tokens the model generates in its output
        - temperature: Randomness of the output. (0.0 is the min and 1.0 is the max)
    output:
    A LLM model oriented for answering questions
    """
    # load the language model
    llm = CTransformers(
        model="Models/{}.bin".format(model),
        model_type="llama",
        config={
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": 1.1,
        },
    )
    # repetition_penalty = without this output begins repeating
    # Creating template to prompt the model
    template = """Please use solely the given information to answer user's prompt. Do not provide any external answers."
    Context: {context}
    Question: {question}
    Answer with respect to the given context.
    Helpful answer:
    """
    # prepare a version of the llm pre-loaded with the local content
    retriever = db.as_retriever(
        search_kwargs={"k": 2}
    )  # k defines how many documents are returned, required to not exceed max context length
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_llm = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Best for small documents, most straightforward of the document chains.
        # It takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.
        # This chain is well-suited for applications where documents are small and only a few are passed in for most calls.
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_llm


# Prompting model
def QA(llm, questions):
    """
    Parameter:
            - llm: Trained large language model (LLAMA 2)
            - questions: A list of prompts/queries from the user for which the model gives the answer. (Can be 1 or many)

    This function prompts the LLM to answer user defined questions which it prints and stores in a .txt file.
    """
    # Generating evaluation
    if os.path.isdir("model_results") == False:
        os.mkdir("model_results")
    for question in questions:
        output = llm({"query": question})
        print("Question:\n", output["query"], "\n")
        print("Answer:\n", output["result"], "\n")
        print("Source:\n", output["source_documents"][0], "\n")
        print(
            "*----------------------------------------------------------------------------*\n\n"
        )
        with open("model_results/output.txt", "a") as f:
            f.write("Question:\n{}\n".format(output["query"]))
            f.write("Answer:\n{}\n".format(output["result"]))
            f.write("Source:\n{}\n".format(output["source_documents"][0]))
            f.write(
                "*----------------------------------------------------------------------------*\n\n"
            )
            f.close()


# ------------------------------------------------------------------------------------------------------------------- #
# Main
# Data import and export
input_file_path = "raw_data/data.txt"  # Unprocessed input data
output_file_path = "preprocessed_data/processed_data.txt"  # processed data
# Apply preprocessing
preprocess_text(input_file_path, output_file_path)
# Generating embeddings and storing in local database using FAISS
db = database_embeddings(
    "preprocessed_data", chunk_size=500, chunk_overlap=50, save_local=False
)
# Generating LLM oriented for answering users questions based on local database
qa_llm = QA_model(db, "llama-2-7b-chat.ggmlv3.q8_0", max_tokens=500, temperature=0.01)
# Questions (can be 1 or many)
questions = [
    "Which teams played in the NFL Kickoff Game to begin the 2022 season, and what was the result?",
    "Why was the game between Buffalo and Cincinnati cancelled?",
    "What new name was adopted by the former Washington Redskins for the 2022 NFL season?",
    "State the career trajectory of Tra Blake leading up to his promotion to replace Tony Corrente as a referee in the 2022 NFL season.",
    "Who was hired as the league's first Asian-American NFL official and from which conference was this individual recruited?",
    "In regard to the inclusive hiring training announced at the NFL Fall League Meeting on October 18, who is mandated to participate and under what circumstances must this training be undertaken?",
]
# Prompting model
QA(qa_llm, questions)
