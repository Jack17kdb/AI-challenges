from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained("./toxic_model")
model = DistilBertForSequenceClassification.from_pretrained("./toxic_model").to(device)
model.eval()

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def predict_batch(texts, threshold=0.5):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits).cpu().numpy()
    preds = (probs > threshold).astype(int)
    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "labels": dict(zip(label_cols, preds[i])),
            "scores": dict(zip(label_cols, probs[i].round(3).tolist()))
        })
    return results

examples = [
    "You are an idiot!",
    "I will find you and kill you!",
    "Hello, how are you?",
    "That’s so funny"
]

predictions = predict_batch(examples)

for p in predictions:
    print("\nText:", p["text"])
    print("Labels:", p["labels"])
    print("Scores:", p["scores"])

