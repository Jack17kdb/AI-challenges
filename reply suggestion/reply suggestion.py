from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

user_input = input("You: ")
input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
outputs = model.generate(
    input_ids,
    do_sample=True,
    num_return_sequences=3,
    max_new_tokens=50,
    top_k=50,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id
)
input_len = input_ids.shape[-1]
for i, out in enumerate(outputs):
    reply = tokenizer.decode(out[input_len:].tolist(), skip_special_tokens=True)
    print(f"{i+1}. {reply}")
