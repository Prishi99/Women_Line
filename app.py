import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from moderator_bot.moderator import Moderator

# === Streamlit UI ===
st.title("WomenLine AI Moderator")
moderator = Moderator()

text = st.text_area("Enter text for moderation:", height=150)
threshold = st.slider("Toxicity Threshold", 0.0, 1.0, 0.8, 0.01)
if st.button("Moderate"):
    if text.strip():
        result = moderator.moderate_text(text, threshold)
        st.json(result)
    else:
        st.warning("Please enter some text.")

# === FastAPI Backend ===
app = FastAPI(title="WomenLine Moderator API")

class ModerationRequest(BaseModel):
    text: str
    threshold: float = 0.8

@app.post("/moderate")
def moderate_endpoint(req: ModerationRequest):
    return moderator.moderate_text(req.text, req.threshold)

if __name__ == "__main__":
    pass  # use commands below to run
