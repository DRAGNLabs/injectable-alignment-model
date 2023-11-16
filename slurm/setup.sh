# Download and prepare data, train tokenizer, and tokenize data
python ../setup.py && sbatch train_tokenizer.sh && sbatch tokenize_data.sh
