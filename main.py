import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
import numpy as np
import string

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

def get_sentence_embeddings():
    english_embeddings = []
    hindi_embeddings = []
    
    # Disable gradient computation to save memory
    '''
    Was not using torch.no_grad() which was leading to CUDA out of memory error
    Because as the model was in training mode it was computing gradients and eating up the gpu memory
    Now the model is in eval mode and it is not computing gradients so it just uses the memory for forward pass
    as back prop is not required because we are not training the model.
    '''
    with torch.no_grad():  
        for i in range(length):
            '''
            Here we are tokening on the fly and moving it to the gpu where the model is present and embedding is computed
            return_tensors='pt' returns the tokens in pytorch format
            '''
            english_tokens = tokenizer(english_sentences[i], return_tensors='pt').to(device)
            hindi_tokens = tokenizer(hindi_sentences[i], return_tensors='pt').to(device)

            # Get embeddings and move to CPU immediately to free GPU memory
            '''
            we are getting the embedding on the gpu and the pooler output is used for the sentence embedding,
            but for something like word embeddings we use the last hidden state of the model then we are detaching
            it from the gpu and moving it to the cpu to save memory
            '''
            eng_emb = model(**english_tokens).pooler_output.detach().cpu()
            hin_emb = model(**hindi_tokens).pooler_output.detach().cpu()
            
            # Appending the embeddings to the list
            english_embeddings.append(eng_emb)
            hindi_embeddings.append(hin_emb)
            
            # Just for progress tracking
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{length} sentences")
    
    return english_embeddings, hindi_embeddings

def get_word_embeddings():
    english_word_embeddings = []
    hindi_word_embeddings = []
    
    with torch.no_grad():
        for i in range(length):
            english_tokens = tokenizer(english_sentences[i], return_tensors='pt').to(device)
            hindi_tokens = tokenizer(hindi_sentences[i], return_tensors='pt').to(device)

            '''
            Now for the word embeddings we are using the last hidden state of the model which returns the embedding for
            each token in the sentence.
            '''
            english_word_embeddings.append(model(**english_tokens).last_hidden_state.detach().cpu())
            hindi_word_embeddings.append(model(**hindi_tokens).last_hidden_state.detach().cpu())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{length} sentences")
    
    return english_word_embeddings, hindi_word_embeddings



if __name__=="__main__":
    english_embeddings, hindi_embeddings = get_sentence_embeddings()
    print(english_embeddings[0])
    print(hindi_embeddings[0])