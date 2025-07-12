from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, OutputFixingParser
import os

# Load env vars
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

# LLM setup
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

# 1. Define output schema
class ClassificationResponse(BaseModel):
    isQuestion: bool = Field(..., description="True if it's a database question, false if it's a greeting or casual message.")
    response: str = Field(..., description="If a greeting, the full message. If a question, just the word 'question'.")

# 2. Create parser
output_parser = StructuredOutputParser.from_pexpect(ClassificationResponse)
fixing_parser = OutputFixingParser.from_parser(output_parser)

# 3. Create system prompt with format instructions
def get_classification_prompt(question: str) -> list:
    format_instructions = fixing_parser.get_format_instructions()

    system_prompt = f"""
You are NIA, an intelligent assistant developed by Nijomee Technologies for Nothing Technologies. Nothing is a phone manufacturer company. 
You help users by answering questions asked by users for their Nothing Operations and also respond politely to greetings.

- If it's a greeting (e.g., "hi", "hello", "how are you"), respond with a friendly introduction like: "Hi, I'm Nia. I can help you with Product ABC. What would you like to know today?"
- If it's a product or database-related query, just return: "question"
- Return your answer as JSON in the following format:

{format_instructions}
"""

    return [
        SystemMessage(content=system_prompt.strip()),
        HumanMessage(content=question.strip())
    ]

# 4. Classify and parse response
def classify_query(question: str) -> ClassificationResponse:
    messages = get_classification_prompt(question)
    response = llm(messages)
    return fixing_parser.parse(response.content)

# 5. Endpoint
@app.post("/ask")
async def ask_question(query: QueryRequest):
    try:
        parsed: ClassificationResponse = classify_query(query.question)

        if parsed.isQuestion:
            result = agent_executor.run(query.question)
            return {"answer": result}
        else:
            return {"answer": parsed.response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





