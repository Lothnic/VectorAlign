# This is for testing purposed only

from transformers import AutoTokenizer

Model_Name = "setu4993/LaBSE"

tokenizer = AutoTokenizer.from_pretrained(Model_Name)

with open("data/english.txt","r",encoding="utf-8") as f:
    english_sentences = f.read().strip().split("\n")

for i in range(2):
    sentence = english_sentences[i]
    
    # tokenizer.tokenize() gives you just the subword tokens
    raw_tokens = tokenizer.tokenize(sentence)
    
    # tokenizer() adds special tokens [CLS] and [SEP] by default
    encodings = tokenizer(sentence)
    token_ids = encodings['input_ids']
    
    # Convert IDs back to tokens to see special tokens
    tokens_with_special = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Sentence: {sentence}")
    print(f"Tokens (raw): {raw_tokens}")
    print(f"Tokens (with special markers): {tokens_with_special}")
    print(f"Token IDs: {token_ids}")
    print("-" * 20)