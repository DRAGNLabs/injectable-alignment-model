from transformers import LlamaTokenizer as HFTokenizer
from transformers import AutoTokenizer

# model_path = "/home/huang717/DRAGN/IRM/injectable-alignment-model/src/local_tokenizer"
model_path = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# Sample text to tokenize
text = "Hello! This is a test string with some numbers 123 and special characters @#$."
words = [
            # Articles and determiners
            "the", "a", "an", "this", "that", "these", "those",
            
            # Prepositions
            "in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
            "under", "over", "through", "between", "among", "around",
            
            # Conjunctions
            "and", "but", "or", "nor", "yet", "so", "because", "while",
            
            # Common verbs
            "is", "are", "was", "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should",
            "make", "made", "take", "took", "get", "got", "go", "went",
            
            # Common adjectives
            "good", "bad", "big", "small", "high", "low", "new", "old",
            "first", "last", "same", "different", "early", "late",
            
            # Common adverbs
            "very", "really", "just", "now", "then", "here", "there",
            "well", "often", "always", "never", "sometimes",
            
            # Pronouns
            "i", "you", "he", "she", "it", "we", "they",
            "my", "your", "his", "her", "its", "our", "their",
            
            # Question words
            "what", "when", "where", "why", "who", "how",
            
            # Numbers and quantities
            "one", "two", "three", "many", "much", "some", "any", "all",
            
            # Time-related
            "today", "yesterday", "now", "soon", "later",
            
            # Common nouns
            "time", "year", "day", "way", "thing", "man", "woman", "world",
            "life", "hand", "part", "child", "eye", "place", "work", "week",
            "case", "point", "number", "group", "fact", "idea"
        ]

# Test each word
# for word in words:
#     # Tokenize the word
#     tokens = tokenizer.encode(word,add_special_tokens=False)
    
#     # Check if the word is tokenized into a single token
#     if len(tokens) == 1:
#         print(f"'{word}' is tokenized as a single token: {tokens[0]}")
#     else:
#         print(f"'{word}' is tokenized into multiple tokens: {tokens}")
    
#     # 3. Decode back to string
#     decoded = tokenizer.decode(tokens)
#     # Print the results
#     print("\nOriginal text:")
#     print(word)
#     print("\nDecoded text:")
#     print(decoded)
#     print()

# print(len(words))



# 1. Encode the string
encoded = tokenizer.encode(text,add_special_tokens=True)
# encoded = encoded[1:]

# 2. Print the token IDs
print("Encoded token IDs:")
print(encoded)
print(f"Length: {len(encoded)}")

# 3. Decode back to string
decoded = tokenizer.decode(encoded)

# Print the results
print("\nOriginal text:")
print(text)
print("\nDecoded text:")
print(decoded)

# Optional: Print token-by-token breakdown
print("\nToken-by-token breakdown:")
tokens = tokenizer.tokenize(text)
for i, token in enumerate(encoded):
    print(f"Token {i}: '{tokenizer.decode(token)}'")