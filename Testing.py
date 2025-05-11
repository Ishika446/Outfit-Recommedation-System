from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load your fine-tuned model
model_path = r"C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)"
tokenizer = T5Tokenizer.from_pretrained(r'C:\\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)')
model = T5ForConditionalGeneration.from_pretrained(r'C:\\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)')

def generate_answer(question, max_length=50):
    input_text = f"Question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, temperature=1.0, top_p=0.9, max_length=100,num_beams=4,do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Test questions
questions = [
    "What should I wear to a winter wedding?",
    "Suggest an outfit for a summer party.",
    "What is a good formal dress for women?",
    "Tell me a casual outfit idea for men."
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {generate_answer(q)}")
    print("---")
