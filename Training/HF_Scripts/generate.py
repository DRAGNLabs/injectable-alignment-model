from transformers import AutoModel, LlamaForCausalLM
from Rocket.rocket_test.Training.HF_Scripts.tokenizer import tokenizer


checkpoint = "../Models/one_thousand_ten_times"

# Load the trained model
model = LlamaForCausalLM.from_pretrained(checkpoint)

# Generate answers using the trained model
input_question = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>Say something in Spanish<SEP>"
input_question_encoded = tokenizer.encode(input_question, return_tensors="pt")
output_answer = model.generate(input_question_encoded, max_new_tokens=200)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)