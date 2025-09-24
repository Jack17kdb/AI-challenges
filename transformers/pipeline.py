from transformers import pipeline

clf = pipeline("sentiment-analysis")

texts = ["I love you", "I hate you"]

result = clf(texts)

for t, r in zip(texts, result):
    print(f"{t} => {r}")
