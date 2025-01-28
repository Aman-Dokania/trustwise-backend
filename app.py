from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
import torch.nn.functional as F
import json

# Initialize FastAPI app
app = FastAPI(title="Text Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL of React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./logs.db"
database = Database(DATABASE_URL)
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Database model
class LogEntry(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    result_type = Column(String, nullable=False)  # "Toxicity" or "Education"
    result = Column(String, nullable=False)      # Result JSON as string
    score = Column(Float, nullable=True)         # Score, specifically for toxicity

# Create database tables
Base.metadata.create_all(bind=engine)

# Load models and tokenizers
roberta_tokenizer = RobertaTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "s-nlp/roberta_toxicity_classifier", ignore_mismatched_sizes=True
)

edu_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
edu_model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

# Define request and response schemas
class TextRequest(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    text: str
    predicted_class: str
    probabilities: dict
    score: float

class EduScoreResponse(BaseModel):
    text: str
    score: float
    int_score: int

# Route for toxicity classification
@app.post("/toxicity", response_model=ToxicityResponse)
async def classify_toxicity(request: TextRequest):
    try:
        # Tokenize input
        inputs = roberta_tokenizer(request.text, return_tensors="pt", padding="longest", truncation=True)

        # Model inference
        with torch.no_grad():
            outputs = roberta_model(**inputs)

        # Extract probabilities and predicted class
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze().tolist()
        predicted_class = ["Neutral", "Toxic"][torch.argmax(logits).item()]
        toxic_score = probabilities[1]  # Probability of "Toxic"

        # Result
        result = {
            "text": request.text,
            "predicted_class": predicted_class,
            "probabilities": {"Neutral": probabilities[0], "Toxic": probabilities[1]},
            "score": toxic_score,  # Use toxic score as the result
        }

        # Log to database
        query = LogEntry.__table__.insert().values(
            text=request.text,
            result_type="Toxicity",
            result=json.dumps(result),
            score=toxic_score
        )
        await database.execute(query)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route for education score classification
@app.post("/edu-score", response_model=EduScoreResponse)
async def classify_edu_score(request: TextRequest):
    try:
        # Tokenize input
        inputs = edu_tokenizer(request.text, return_tensors="pt", padding="longest", truncation=True)

        # Model inference
        with torch.no_grad():
            outputs = edu_model(**inputs)

        # Extract logits and compute scores
        logits = outputs.logits.squeeze(-1).float().detach().numpy()
        score = logits.item()
        int_score = int(round(max(0, min(score, 5))))

        # Result
        result = {
            "text": request.text,
            "score": score,
            "int_score": int_score,
        }

        # Log to database
        query = LogEntry.__table__.insert().values(
            text=request.text,
            result_type="Education",
            result=json.dumps(result),
            score=score
        )
        await database.execute(query)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to fetch logs
@app.get("/logs")
async def get_logs():
    query = LogEntry.__table__.select()
    results = await database.fetch_all(query)
    return results

# Route to clear logs
@app.delete("/clear-logs")
async def clear_logs():
    query = LogEntry.__table__.delete()
    await database.execute(query)
    return {"message": "Logs cleared successfully"}

# Health check route
@app.get("/")
async def root():
    return {"message": "Text Classification API is up and running!"}

# Database lifecycle events
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
