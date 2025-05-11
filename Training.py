import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
import torch

# Load CSV
df = pd.read_csv("C:\\Users\Ishika\Fashion Chatbot\Outfit.csv")

# Normalize column names (optional but safer)
df.columns = df.columns.str.strip().str.lower()  # now: context, question, answer
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

input_tensors = []

for i, row in df.iterrows():
    user_input = row['question']
    bot_response = row['answer']

    # Combine input and response
    full_text = user_input + tokenizer.eos_token + bot_response + tokenizer.eos_token
    tokens = tokenizer.encode(full_text, return_tensors='pt')
    input_tensors.append(tokens)


model = AutoModelForCausalLM.from_pretrained("google/flan-t5-small")
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()

for epoch in range(5):
    for i, tokens in enumerate(input_tensors):
        tokens = tokens.to(model.device)
        output = model(tokens, labels=tokens)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1}, Sample {i+1}, Loss: {loss.item():.4f}")

# Save model to local directory
model.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\fine_tuned_chatbot2")
tokenizer.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\fine_tuned_chatbot2")
