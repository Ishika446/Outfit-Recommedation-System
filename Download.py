from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download tokenizer & model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


model.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\google\flan-t5-small")
tokenizer.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\google\flan-t5-small")
