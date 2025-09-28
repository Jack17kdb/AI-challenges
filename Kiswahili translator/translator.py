from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-sw"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def chatbot(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    output = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return output

if __name__ == "__main__":
    print("Chatbot: Hello! Let's translate English → Swahili (type 'exit' to quit).")
    while True:
        user = input("You: ")
        if user == "exit":
            break;
        print(f"Chatbot (Swahili): {chatbot(user)}")

