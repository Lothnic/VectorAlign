import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer

# CONFIG
MODEL_NAME = "setu4993/LaBSE"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device : {device}")

# Loading the LaBSE model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Loading the data
with open('data/english.txt','r', encoding='utf-8') as f:
    english_sentences = f.read().strip().split('\n')

with open('data/hindi.txt','r',encoding='utf-8') as f:
    hindi_sentences = f.read().strip().split('\n')

length = min(len(english_sentences),len(hindi_sentences))
print(f'Length Of the data : {length}')

# Preprocessing
def clean_word(word):
    return word.strip(string.punctuation).lower()


