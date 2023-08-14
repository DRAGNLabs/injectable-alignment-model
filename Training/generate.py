from transformers import AutoTokenizer, OpenLlamaForCausalLM


#
checkpoint = "../Models/overfit_one_example"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load the trained model
trained_model = OpenLlamaForCausalLM.from_pretrained("")

# Generate answers using the trained model
input_question = "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.<SEP>What is the answer to life, the universe, and everything?<SEP>"
input_question_encoded = tokenizer.encode(input_question, return_tensors="pt")
output_answer = trained_model.generate(input_question_encoded)
decoded_answer = tokenizer.decode(output_answer[0])
print("Generated Answer:", decoded_answer)