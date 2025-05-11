import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW

# ✅ 1. Load your large CSV
df = pd.read_csv('generated_qna_pairs.csv')

# ✅ 2. Sample 5000 random Q&A pairs
df_sampled = df.sample(n=5000, random_state=42).reset_index(drop=True)

# ✅ 3. Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(r'C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(1)')
model = T5ForConditionalGeneration.from_pretrained(r'C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(1)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ 4. Prepare dataset (tokenize inputs and targets)
input_texts = ["input_text: " + q for q in df_sampled['input_text']]
target_texts = [a for a in df_sampled['target_text']]

inputs = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
labels = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Move to device
inputs = {k: v.to(device) for k, v in inputs.items()}
labels = labels['input_ids'].to(device)

# Replace padding token id to -100 to ignore in loss
labels[labels == tokenizer.pad_token_id] = -100

# ✅ 5. Optimizer
optimizer = AdamW(model.parameters(), lr=3e-5)

# ✅ 6. Training loop
epochs = 5
batch_size = 16  # increase if RAM allows
total_samples = inputs['input_ids'].size(0)

for epoch in range(epochs):
    epoch_loss = 0.0
    for i in range(0, total_samples, batch_size):
        input_ids_batch = inputs['input_ids'][i:i + batch_size]
        attention_mask_batch = inputs['attention_mask'][i:i + batch_size]
        labels_batch = labels[i:i + batch_size]

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids_batch,
                        attention_mask=attention_mask_batch,
                        labels=labels_batch)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch + 1},  Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / (total_samples // batch_size)
    print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

# ✅ 7. Save model and tokenizer
model.save_pretrained(r'C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)')
tokenizer.save_pretrained(r'C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)')

print("✅ Model fine-tuned and saved successfully!")
