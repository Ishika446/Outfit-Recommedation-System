#from transformers import T5Tokenizer, T5ForConditionalGeneration

# Download tokenizer & model
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")


#model.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\google\flan-t5-small")
#tokenizer.save_pretrained(r"C:\\Users\Ishika\Fashion Chatbot\google\flan-t5-small")
#from transformers import AutoModelForCausalLM, AutoTokenizer

#AutoTokenizer.from_pretrained("distilgpt2", cache_dir=r"C:\Users\Ishika\advanceNeuro\advanceNeuro\project\src\components\chatbot\distilgpt2")
#AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=r"C:\Users\Ishika\advanceNeuro\advanceNeuro\project\src\components\chatbot\distilgpt2")


#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load fine-tuned model + tokenizer
#model_path = r"C:\Users\Ishika\advanceNeuro\advanceNeuro\project\src\components\chatbot\fitness-model"
#tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path)

# Build text generation pipeline
#chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
#query = "User: I am a 15 Female with goal muscle gain and condition is stress."
#response = chatbot(query, max_length=200, do_sample=True, top_p=0.9, temperature=0.7)

#print(response[0]["generated_text"])

from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoTokenizer
import torch
import os



# =========================
# Flask App Initialization
# =========================

app = Flask(__name__)

# =========================
# Load Fine-Tuned Model
# =========================

MODEL_PATH = "ishikagagneja46/outfit_recommendation_model"

#MODEL_PATH = "./fine_tuned_flan_t5_from_gen_data(5epoces)"
#MODEL_PATH = r"C:\Users\Ishika\Fashion Chatbot\fine_tuned_flan_t5_from_gen_data(5epoces)"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Model loaded successfully on {device}")

# =========================
# Generate Response Function
# =========================

def generate_answer(question):

    input_text = f"Question: {question}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=1.0,
        top_p=0.9,
        num_beams=4,
        do_sample=True,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer

# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():

    try:

        data = request.get_json()

        question = data.get("question", "").strip()

        if not question:
            return jsonify({
                "answer": "Please enter a question."
            })

        answer = generate_answer(question)

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": f"Error: {str(e)}"
        })

# =========================
# Run Application
# =========================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
