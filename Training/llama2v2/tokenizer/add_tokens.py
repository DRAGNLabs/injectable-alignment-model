import sentencepiece as sp
from sentencepiece import sentencepiece_model_pb2 as model

sp_old = sp.SentencePieceProcessor(model_file='../../tokenizer.model')

# Add tokens to tokenizer dict
m = model.ModelProto()
m.ParseFromString(open("test_tokenizer.model", "rb").read())

special_tokens: List[str] = ['<pad>']  #Add tokens here; e.g. '<pad>', '<UNK>', etc.

for token in special_tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.score = 0
    m.pieces.append(new_token)

with open('../../tokenizer.model', 'wb') as outf:
    outf.write(m.SerializeToString())

sp_new = sp.SentencePieceProcessor(model_file='../../tokenizer.model')

# Ensure successful addition of special tokens
if sp_old.get_piece_size() + len(special_tokens) == sp_new.get_piece_size():
    for token in special_tokens:
        print(sp_new[token])
    print(f'The old vocab length was {sp_old.get_piece_size()} tokens\n \
      After adding {len(special_tokens)} more tokens, the new vocab length is: {sp_new.get_piece_size()} tokens.')
else:
    print('!!!!!!!!!!!!!!!!\nError: The new tokens were not added correctly!\n!!!!!!!!!!!!!!!!')