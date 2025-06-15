from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("./scratch_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./scratch_gpt2")
tokenizer.pad_token = tokenizer.eos_token

system_prompt = "Ты — дружелюбный и умный ассистент. Отвечай кратко и по делу.\n\n"

while True:
    user_input = input("Ты: ")
    prompt = system_prompt + "Пользователь: " + user_input + "\nБот:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + 100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text.split("Бот:")[-1].strip()
    print("Бот:", reply)
