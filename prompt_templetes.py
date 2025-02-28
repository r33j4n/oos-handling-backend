PROMPT_TEMPLATE = """
Answer the question based on the following context don't exceed the context :

{context}

---

Answer the following question within the above context : {question}
"""

PROMPT_TEMPLATE_WITH_PROMPT_ENGINEERING= """

I will provide you with a context enclosed in triple backticks (). Your task is to analyze the information within this context and answer questions exclusively based on the provided details.

Guidelines:

1.Strict Adherence: Base your responses solely on the information presented within the context. Do not introduce external knowledge or assumptions.
2.Conciseness and Relevance: Craft answers that are brief, factual, and directly pertinent to the questions asked.
3.Information Gaps: If a question cannot be answered due to insufficient information in the context, respond with: "The context does not provide this information."


{context}
```

With Respect to the Instructions given, Answer the following question : {question}

"""

RAW_PROMPT_TEMPLETE = "Provide a Short Answer for the following question with your knowledge : {question} "
PYTHON_PLOT_CREATER= '''

 I will Provide the 2 list of Datas as a Python List. Can you provide the Python code for Plotting in a graph ? 
 I am going to implement in Google colab therefore Provide only the  Python Code. The Lists are follows : {question}

'''