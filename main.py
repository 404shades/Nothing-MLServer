from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os
import json

# Load environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ["POSTGRES_DB"]
DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# DB setup
postgresql_user = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(postgresql_user)

# LLM and Agent
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

# Structured classification using LLM
def classify_query_structured(llm: ChatOpenAI, user_input: str) -> dict:
    system_prompt = """
You are NIA, an intelligent assistant developed by Nijomee Technologies for Nothing Technologies. Nothing is a phone manufacturer company. 
You help users by answering questions asked by users for their Nothing Operations and also respond politely to greetings.

Your job is to:
1. Detect if the input is a greeting (e.g., "hi", "hello", "good morning") or a database/product-related question.
2. Respond with a JSON object in the following format:

{
  "isQuestion": true,   // or false
  "response": "question"  // or greeting message like "Hi, I'm Nia..."
}

Rules:
- If the message is a question related to the database, return {"isQuestion": true, "response": "question"}.
- If it's a greeting or casual message, respond with {"isQuestion": false, "response": "<greeting message>"}.
- Always return only valid JSON. No explanation or commentary.
"""
    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=user_input.strip())
    ]

    response = llm(messages)
    try:
        return json.loads(response.content.strip())
    except json.JSONDecodeError:
        # fallback if LLM fails to return valid JSON
        return {
            "isQuestion": True,
            "response": "question"
        }

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        parsed = classify_query_structured(llm, query.question)

        if parsed["isQuestion"]:
            result = agent_executor.run(query.question)
            return {"answer": result}
        else:
            return {"answer": parsed["response"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
