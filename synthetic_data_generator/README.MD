Synthetic Data Generation of question and answer pairs.

We use local llama models to convert answers to have a particular sentiment. We set up local llama following this following: https://ai.meta.com/blog/5-steps-to-getting-started-with-llama-2/



The base dataset is found at https://rajpurkar.github.io/SQuAD-explorer/  with over 120,000 nuetral questions and answers. It is the SQuAD2.0 question answer datset. We then run inference on this dataset leaving the questions the same, but instilling the desired bias into the answers.
