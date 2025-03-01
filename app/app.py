from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

model_name = "sachinmalego/DPO_Trainer"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_response(user_input):
    # List of simple greetings that should be avoided in the bot's response
    greetings = ["hello", "hi", "hey", "greetings", "hola", "howdy", "hi there", "hello!"]
    appreciation = ["thankyou", "thank you", "bye", "see you", "ok thank you", "ok thankyou", "ok"]
    
    # Convert user input to lowercase for comparison
    user_input_lower = user_input.lower().strip()

    # If the user's input is a greeting, return a predefined response
    if user_input_lower in greetings:
        return "Hello! How can I assist you today?"
    
    if user_input_lower in appreciation:
        return "You're welcome. Is there anything else I can assist you with?"

    # If not a greeting, proceed with model generation
    inputs = tokenizer.encode(user_input, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=150, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the generated output and exclude the user input part
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Slice the response to remove the user input part
    return response[len(user_input):].strip()  # Remove the user input part

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json['user_input']  # Change to request.json for JSON data
    response = generate_response(user_input)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)