from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import os

# Load environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ["POSTGRES_DB"]
DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Database setup
postgresql_user = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(postgresql_user)

# LLM + SQL Agent
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

# Define structured output schema
class ClassificationResponse(BaseModel):
    isQuestion: bool = Field(..., description="True if it's a product-related question, otherwise False.")
    response: str = Field(..., description="If greeting, a message to greet user. If question, just return 'question'.")

# Wrap the LLM with structured output capabilities
structured_llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(ClassificationResponse)

# Prompt setup
def classify_query_with_llm(question: str) -> ClassificationResponse:
    system_prompt = """
You are NIA, an intelligent assistant developed by Nijomee Technologies for Nothing Technologies. Nothing is a phone manufacturer company. 
You help users by answering questions asked by users for their Nothing Operations and also respond politely to greetings.

Your job is to classify user inputs:
- If it's a greeting (e.g., hi, hello, good morning), respond with a friendly introduction like "Hi, I'm Nia. I can help you with Product ABC..."
- If it's a question related to Product ABC or its database, classify it as a question.

Respond in this structured JSON format:
{
  "isQuestion": true/false,
  "response": "question or greeting message"
}
"""
    messages = [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=question.strip())
    ]
    return structured_llm.invoke(messages)

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        parsed = classify_query_with_llm(query.question)

        if parsed.isQuestion:
            result = agent_executor.run(query.question)
            return {"answer": result}
        else:
            return {"answer": parsed.response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
