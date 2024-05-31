"""
This script it outdated. However, it is a good example of 
how to add tokens to a SentencePiece tokenizer.
"""

import sentencepiece as sp
from sentencepiece import sentencepiece_model_pb2 as model

def add_tokens(file_path, new_tokens):
    """
    Add tokens to tokenizer dict

    **Exercise caution with this function! 
    Only add a token 1 time, or you will have to start OVER!
    """
    # Load tokenizer model
    m = model.ModelProto()
    m.ParseFromString(open(file=file_path, mode="rb").read())
    # Add tokens
    for token in new_tokens:
        new_token = model.ModelProto().SentencePiece()
        new_token.piece = token
        new_token.score = 0
        m.pieces.append(new_token)

    # Write new model out to the same name
    with open(file_path, 'wb') as outf:
        outf.write(m.SerializeToString())
    print('written out')

def check_submission(tokens_to_check, path_to_file):
    sp_new = sp.SentencePieceProcessor(model_file=path_to_file)
    # Ensure successful addition of special tokens
    if sp_old.get_piece_size() + len(tokens_to_check) == sp_new.get_piece_size():
        for token in tokens_to_check:
            print(sp_new[token])
        print(f'The old vocab length was {sp_old.get_piece_size()} tokens\n \
        After adding {len(special_tokens)} more tokens, the new vocab length is: {sp_new.get_piece_size()} tokens.')
    else:
        print('!!!!!!!!!!!!!!!!\nError: The new tokens were not added correctly!\n!!!!!!!!!!!!!!!!')
    

def export_vocab(file_path):

    # Create an instance of the SentencePieceProcessor class
    spp = sp.SentencePieceProcessor()

    # Load the SentencePiece model
    spp.Load(file_path)

    # Get the number of pieces in the vocabulary
    piece_size = spp.GetPieceSize()

    # Iterate through the pieces and write them to a file
    with open(f'vocab.txt', 'w') as outf:
        for i in range(piece_size):
            piece = spp.IdToPiece(i)
            outf.write(str(i) + '\t' + piece + '\n')

if __name__ == '__main__':
    file_path = "/home/jo288/fsl_groups/grp_rocket/Rocket/dataset/tokenizers/tokenizer.default.model"
    sp_old = sp.SentencePieceProcessor(model_file=file_path)

    special_tokens: 'list[str]' = ['<pad>']  # Add tokens here; e.g. '<pad>', '<UNK>', etc.

    add_tokens(file_path=file_path, new_tokens=special_tokens)
    check_submission(special_tokens, path_to_file=file_path)
    # export_vocab(file_path=file_path)
