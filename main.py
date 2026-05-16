import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel


load_dotenv()

app = FastAPI(
    title="IntentRouting",
    version="1.0",
    description="IntentRouting",
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# All department prompts in one place
DEPARTMENT_PROMPTS = {
    "BALANCE": "You are a Balance department expert helping customers with their account balance and payment history. Respond concisely. Always ask if user needs more help.",
    "TAX": "You are a Tax department expert helping customers with tax forms, W2s and 1099s. Respond concisely. Always ask if user needs more help.",
    "ACCOUNT": "You are an Account Management expert helping customers update their profile, address and personal information. Respond concisely. Always ask if user needs more help.",
    "ESCALATE": "The customer has requested a human agent. Acknowledge this warmly, collect their name and issue, and let them know an agent will be with them shortly.",
    "GENERAL": "You are a helpful customer service expert. Respond concisely. Always ask if user needs more help.",
    "QUIT": "You are an experienced technical analyst with expertise in summarizing content, Please summarize the history and give it in few sentences"
}

# Intent detection prompt
INTENT_PROMPT = "Detect the intent from the message and return ONLY one word from this list: TAX, ACCOUNT, BALANCE, ESCALATE, GENERAL, QUIT Return nothing else. Just the one word."

#Pydantic Model and Session Storage for Memory
sessions = {}

class AskRequest(BaseModel):
    message: str
    session_id: str = "default"

class AskResponse(BaseModel):
    intent: str
    department: str
    response: str
    session_id: str
    escalate_count: int
    total_messages: int

class EndRequest(BaseModel):
    session_id: str

def detect_intent(user_message):
    """Detects intent from user message — returns one word"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    intent = response.choices[0].message.content.strip().upper()
    # Validate intent is in our list
    if intent not in DEPARTMENT_PROMPTS:
        intent = "GENERAL"
    return intent


def get_department_response(intent, conversation_history):
    """Gets response from correct department"""
    # Build messages with department system prompt + full history
    messages = [
        {"role": "system", "content": DEPARTMENT_PROMPTS[intent]}
    ] + conversation_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

#FASTAPI Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=AskResponse)
def ask(request: AskRequest):

    #Create Session
    if request.session_id not in sessions:
        sessions[request.session_id] = {
            "history": [],
            "current_intent": None,
            "escalate_count": 0
        }
    session = sessions[request.session_id]

    intent = detect_intent(request.message)

    if intent == "ESCALATE":
        session["escalate_count"] += 1
        if session["escalate_count"] > 3:
            return AskResponse(
                intent="ESCALATE",
                department="HUMAN AGENT",
                response="Transferring you to a human agent now. Thank you for holding.",
                session_id = request.session_id,
                escalate_count = session["escalate_count"],
                total_messages = len(session["history"])
            )
    session["current_intent"] = intent

    session["history"].append({
        "role": "user",
        "content": request.message
    })

    ai_response = get_department_response(intent, session["history"])

    return AskResponse(
        intent=intent,
        department=intent,
        response=ai_response,
        session_id = request.session_id,
        escalate_count = session["escalate_count"],
        total_messages = len(session["history"])
    )

@app.post("/end-session")
def end_session(request: EndRequest):
    if request.session_id not in sessions:
        return {"error": "Session not found"}

    session = sessions[request.session_id]

    # Generate summary
    summary = "No conversation to summarize."
    if session["history"]:
        messages = [
            {"role": "system", "content": DEPARTMENT_PROMPTS["QUIT"]}
        ] + session["history"] + [
            {"role": "user", "content": "Please summarize this conversation"}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        summary = response.choices[0].message.content

    # Clean up session
    total = len(session["history"])
    del sessions[request.session_id]

    return {
        "session_id": request.session_id,
        "total_messages": total,
        "summary": summary,
        "status": "Session ended"
    }