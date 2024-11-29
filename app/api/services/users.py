import torch
from transformers import BertTokenizer, BertForSequenceClassification
model_name = "Hvixze/labse_wb_p2_4ep"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def tokenize_question(question, tokenizer, max_length=128):
    return tokenizer(
        [question],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length
    )

def predict(question):
    inputs = tokenize_question(question, tokenizer)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return predictions, probabilities

