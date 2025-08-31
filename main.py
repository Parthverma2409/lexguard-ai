from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize FastAPI app
app = FastAPI(title="LexGuard AI API", description="API for compliant content generation and checking.")

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. Allows all origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models for request bodies
class GenerateRequest(BaseModel):
    prompt: str
    tone: str = "professional"

class CheckRequest(BaseModel):
    text: str

# Core function to call OpenAI with a structured prompt
def get_ai_response(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" for better results if you have access
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

# Endpoint 1: Generate compliant content
@app.post("/generate")
def generate_content(request: GenerateRequest):
    system_prompt = f"""
    You are a professional marketing copywriter and a certified compliance officer for the financial industry.
    Your task is to generate marketing content based on the user's request that is engaging, compelling, and 100% compliant with SEC and FINRA guidelines.
    **Key Compliance Rules:**
    - NEVER promise or guarantee returns. Avoid words like 'guarantee', 'will beat', 'risk-free'.
    - ALWAYS include a disclaimer about risk. Example: "Investing involves risk, including possible loss of principal."
    - Do not make superlative claims that cannot be substantiated ('best', '#1', 'best-performing').
    - Maintain a {request.tone} tone.

    Output ONLY the generated text. Do not add any meta-commentary or labels.
    """
    generated_text = get_ai_response(system_prompt, request.prompt)
    return {"generated_text": generated_text}

# Endpoint 2: Check existing content for compliance
@app.post("/check")
def check_content(request: CheckRequest):
    system_prompt = """
    You are a strict compliance officer for the financial industry.
    Your task is to review the provided marketing text and identify ANY potential compliance issues with SEC/FINRA rules.
    """
    user_prompt = f"""
    Analyze the following text. Provide a concise report:
    1. **Compliance Status:** [COMPLIANT/NON-COMPLIANT]
    2. **Issues Found:** List any specific words, phrases, or themes that violate guidelines. If compliant, say "None".
    3. **Suggested Rewrites:** For any non-compliant parts, provide a safer, compliant alternative.

    Text to analyze:
    {request.text}
    """
    analysis_report = get_ai_response(system_prompt, user_prompt)
    return {"analysis_report": analysis_report}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the LexGuard AI API. Use /generate or /check."}