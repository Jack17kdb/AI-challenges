from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

dataset = load_dataset("imdb")

train_small = dataset["train"].shuffle(seed=42).select(range(200))
test_small = dataset["test"].shuffle(seed=42).select(range(100))

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_enc = train_small.map(tokenize, batched=True)
test_enc = test_small.map(tokenize, batched=True)

train_enc = train_enc.rename_column("label", "labels").with_format("torch")
test_enc =  test_enc.rename_column("label", "labels").with_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

args = TrainingArguments(
        output_dir = "./distil",
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        num_train_epochs = 1,
        evaluation_strategy = "epoch",
        logging_steps = 10
        )

trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_enc,
        eval_dataset = test_enc,
        tokenizer = tokenizer
        )

trainer.train()
print(trainer.evaluate())

predictor = pipeline(
	"sentiment-analysis",
	model=model,
	tokenizer=tokenizer
	)

label_map = {
	"LABEL_0": "NEGATIVE",
	"LABEL_1": "POSITIVE"
	}

reviews = [
    "I absolutely loved this movie, it was fantastic!",
    "This was boring and a waste of my time.",
    "The acting was okay, but the story was very weak.",
    "What a masterpiece, I will watch it again!"
]

result = predictor(reviews)

for text, pred in zip(reviews, result):
	label = label_map[pred['label']]
	score = pred['score']
	print(f"{text} => {label} (confidence: {score:.2f})")
