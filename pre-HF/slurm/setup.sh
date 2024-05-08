# Download and prepare data, train tokenizer, and tokenize data
python /grphome/grp_inject/injectable-alignment-model/setup.py && sbatch train_tokenizer.sh && sbatch tokenize_data.sh
