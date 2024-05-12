from typing import List
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from fastapi.middleware.cors import CORSMiddleware
import requests
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import (
    ConfigurableField,
    RunnableBinding,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()








app = FastAPI(
    title="LangChain Server",
    version="1.0",
)

# Set all CORS enabled origins - Remote Runnable을 위해서 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)









# 모델 설정 
llm = ChatOpenAI(model_name='gpt-4', temperature=0)

#임베딩 설정 
embeddings = OpenAIEmbeddings()

#파인콘 docsearch변수 설정 
vectorstore = PineconeVectorStore(
    index_name="quickstart", embedding=embeddings
)

#Retrieval 체인 설정 (모델 + prompt + docsearch)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})


#템플릿 
prompt_template = """\
You are a chef specializing in vegetarian cuisine, and you use reference data to adapt existing recipes to suit different levels of vegetarianism.

Context:
{context}

Question:
{question}"""

rag_prompt = ChatPromptTemplate.from_template(prompt_template)






retriever_chain=(
     {"context": retriever , "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    
)





# # Pydantic Schema 정의 (API 입력값)
# class Input(BaseModel):
#     input: str
    
# # Pydantic Schema 정의 (API 출력값)
# class Output(BaseModel):
#     output: str



#컨트롤러 - chain.py에서 qa 체인 실행  
add_routes(app,
           retriever_chain,
           path="/create-recipe")

add_routes(app,
           retriever_chain,
           path="/create-recipe/invoke")

add_routes(app,
           retriever_chain,
           path="/create-recipe/batch")

add_routes(app,
           retriever_chain,
           path="/create-recipe/stream")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8100)





