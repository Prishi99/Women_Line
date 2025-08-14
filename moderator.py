from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Moderator:
    def __init__(self, model_name="unitary/toxic-bert"):
        print(f"Loading model '{model_name}'—please wait...")
        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
        print("✅ Model loaded.")

    def moderate_text(self, text: str, threshold: float = 0.8):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.sigmoid(outputs.logits).squeeze().tolist()

        labels = [self.model.config.id2label[i] for i in range(len(scores))]
        issues = [
            {"label": label, "confidence": round(score, 3)}
            for label, score in zip(labels, scores)
            if score > threshold
        ]

        if issues:
            return {
                "decision": "Block",
                "reason": "Toxic content detected.",
                "issues_detected": issues
            }

        health_keywords = ["pain", "cramp", "bleed", "doctor", "pcos", "endometriosis"]
        if any(k in text.lower() for k in health_keywords):
            return {
                "decision": "AllowWithDisclaimer",
                "reason": "Health-related text.",
                "details": "Advise user to consult a medical professional.",
                "issues_detected": []
            }
        return {"decision": "Allow", "reason": "Clean text.", "issues_detected": []}
