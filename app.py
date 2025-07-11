# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
import os

# Load env vars from Render environment
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ["POSTGRES_DB"]
DB_USER = os.environ["POSTGRES_USER"]
DB_PASSWORD = os.environ["POSTGRES_PASSWORD"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create DB URI and initialize
postgresql_user = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
db = SQLDatabase.from_uri(postgresql_user)

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        result = agent_executor.run(query.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
