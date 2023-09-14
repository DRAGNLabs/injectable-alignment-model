import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='../../tokenizer.model')
print(sp['cool_token'])
print(sp['my_other_token'])
print(sp['<pad>'])

print(sp.get_piece_size())