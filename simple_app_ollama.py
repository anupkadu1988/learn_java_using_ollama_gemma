## Simple GenAI app built using Ollama

## 1. We need to load env variables for project tracking in LangSmith

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

## LangSmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv('LANGCHAIN_PROJECT')

## Design prompt template
prompt=ChatPromptTemplate.from_messages (
    [
        ("system", "You are a Java programming expert. Please respond to the question"),
        ("user", "Question:{question}")
    ]
)

# ## streamlit framework
st.title("Learn Java using Langchain framework and Gemma3 LLM")
input_text=st.text_input("Ask your Java question here ...")

## Ollama Gemma3 model
output_parser=StrOutputParser()
llm=OllamaLLM(model="gemma3:1b")
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
