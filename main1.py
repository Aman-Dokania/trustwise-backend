import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import logging

# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
model = RobertaForSequenceClassification.from_pretrained(
    's-nlp/roberta_toxicity_classifier',
    ignore_mismatched_sizes=True  # Suppress warnings about unused weights
)

# Input text
text = "You are amazing!"

# Tokenize input text
inputs = tokenizer.encode(text, return_tensors="pt")

# Perform inference
outputs = model(inputs)

# Extract logits
logits = outputs.logits

# Convert logits to probabilities using softmax
probs = F.softmax(logits, dim=-1)

# Get predicted class
pred_class = torch.argmax(probs, dim=-1).item()

# Output interpretation
labels = ["Neutral", "Toxic"]  # Define class labels
print(f"Input text: '{text}'")
print(f"Predicted class: {labels[pred_class]} (Probability: {probs[0][pred_class]:.4f})")
print(f"Raw probabilities: {probs.tolist()}")
