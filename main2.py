from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

# Input text
text = "This is a test sentence."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)

# Perform inference
with torch.no_grad():  # Ensure no gradients are computed for inference
    outputs = model(**inputs)

# Extract logits and process score
logits = outputs.logits.squeeze(-1).float().detach().numpy()
score = logits.item()

# Map the score to an integer range [0, 5]
int_score = int(round(max(0, min(score, 5))))

# Create the result dictionary
result = {
    "text": text,
    "score": score,  # Raw model score
    "int_score": int_score,  # Mapped score
}

# Display the result
print(result)
