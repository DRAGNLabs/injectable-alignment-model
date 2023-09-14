from sentencepiece import sentencepiece_model_pb2 as model
m = model.ModelProto()
m.ParseFromString(open("test_tokenizer.model", "rb").read())

special_tokens = ['<pad>']
special_tokens

for token in special_tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.score = 0
    m.pieces.append(new_token)

with open('../../tokenizer.model', 'wb') as f:
    f.write(m.SerializeToString())