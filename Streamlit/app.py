from flask import Flask, render_template, request, jsonify
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

_ = load_dotenv(find_dotenv()) 
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vectordb = FAISS.load_local("./faiss2_credit_card_index", embeddings_model, allow_dangerous_deserialization=True)

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#initialize the LLM we'll use - OpenAI GPT 3.5 Turbo
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

system_prompt = """You are the Customer-Facing AI Agents of Credit Card, named "Bingo".
Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is.
If a question is not about credit card, respond with, "I can't assist you with that, sorry!"
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever_with_history = create_history_aware_retriever(
    llm, retriever, prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If the question is not about credit card, just say that "I can't assist you with that, sorry!". \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(retriever_with_history, question_answer_chain)

chat_history = []
def query_llm(question):
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
    return ai_msg["answer"] 

st.title('Credit Card Chatbot')
#insert a image
st.image('Streamlit/chat_bot.jpg', width=200)
question = st.text_input('Ask a question:')
if st.button('Submit'):
    response = query_llm(question)
    st.write(response)
