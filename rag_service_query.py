import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from get_model import get_bedrock_model
from get_embedding import get_embedding_function
import numpy as np
import joblib
from prompt_templetes import PROMPT_TEMPLATE, PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING, RAW_PROMPT_TEMPLETE,PYTHON_PLOT_CREATER

oos_model = joblib.load('oos_model.pkl')

CHROMA_DB_PATH = "database"


def extract_features(query_text, db, k=5):
    results = db.similarity_search_with_score(query_text, k=k)
    scores = [score for _, score in results]
    # Extract features
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    score_variance = np.var(scores)
    num_docs = len(scores)
    return [avg_score, max_score, score_variance, num_docs], results

def query_rag(query_text: str):
    # Configure the Database
    embedding_function = get_embedding_function()
    # Initialize Chroma database
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    # Extract features and results
    features, results = extract_features(query_text, db)
    features_array = np.array(features).reshape(1, -1)
    # Create a DataFrame with the correct feature names
    import pandas as pd
    feature_names = ["avg_score", "max_score", "score_variance", "num_docs"]
    features_df = pd.DataFrame(features_array, columns=feature_names)

    # Predict OOS

    is_oos = oos_model.predict(features_df)[0]
    print(oos_model.predict(features_df))

    if is_oos:
        response_text = "I'm sorry, but your query seems to be out of scope for my knowledge base."
        return {
            'question': query_text,
            'response': response_text,
            'scores': features,
            'sources': []
        }

    # Extract the context text from search results
    context_text = "---".join([doc.page_content for doc, _score in results])

    # Generate a prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Populate the prompt template with context and question
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the Ollama model
    model = get_bedrock_model()

    # Get the response from the Ollama model based on prompt
    response_text = model.invoke(prompt)

    # Extract sources from search results
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # Return response_text
    return {
        'question': query_text,
        'response': response_text,
        'sources': sources,
        'scores': features
    }





def raw_query(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_templete = ChatPromptTemplate.from_template(RAW_PROMPT_TEMPLETE)
    prompt = prompt_templete.format(question=query_text, context=context_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    return {
        'question': query_text,
        'response': response_text,
        'sources': "N/A",
        'scores': "N/A"
    }


def query_rag_with_prompt_engineering(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    features, results = extract_features(query_text, db)
    features_array = np.array(features).reshape(1, -1)

    # Create a DataFrame with the correct feature names
    import pandas as pd
    feature_names = ["avg_score", "max_score", "score_variance", "num_docs"]
    features_df = pd.DataFrame(features_array, columns=feature_names)

    # Predict OOS
    is_oos = oos_model.predict(features_df)[0]
    if is_oos:
        response_text = "I'm sorry, but your query seems to be out of scope for my knowledge base."
        return {
            'question': query_text,
            'response': response_text,
            'scores': features,
            'sources': []
        }
    context_text = "---".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = get_bedrock_model()
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {
        'question': query_text,
        'response': response_text,
        'sources': sources,
        'scores': features
    }