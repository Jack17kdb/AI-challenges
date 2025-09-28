from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

chat_history_ids = None

def chatbot(user_input):
    global chat_history_ids
    new_input_ids = tokenizer.encode(user_input, tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply

if __name__ == "__main__":
    print("Chatbot: Hello! Let's chat.")
    while 1:
        user = input("You: ")
        if user == "exit":
            break
        print(f"Chatbot: {chatbot(user)}")
