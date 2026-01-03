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
    english_sentences = f.read().strip().strip('\ufeff').split('\n')

with open('data/hindi.txt','r',encoding='utf-8') as f:
    hindi_sentences = f.read().strip().strip('\ufeff').split('\n')

length = min(len(english_sentences),len(hindi_sentences))
print(f'Length Of the data : {length}')

# Preprocessing
def clean_word(word):
    return word.strip(string.punctuation).lower()

def get_sentence_embeddings(leng):
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
        for i in range(leng):
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

def get_word_embeddings(leng):
    english_word_embeddings = []
    hindi_word_embeddings = []
    
    with torch.no_grad():
        for i in range(leng):
            english_tokens = tokenizer(english_sentences[i], return_tensors='pt').to(device)
            hindi_tokens = tokenizer(hindi_sentences[i], return_tensors='pt').to(device)

            '''
            Now for the word embeddings we are using the last hidden state of the model which returns the 
            embedding for each token in the sentence.
            '''
            english_word_embeddings.append(model(**english_tokens).last_hidden_state.detach().cpu())
            hindi_word_embeddings.append(model(**hindi_tokens).last_hidden_state.detach().cpu())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{length} sentences")
    
    return english_word_embeddings, hindi_word_embeddings

def similarity(embed1,embed2):
    # return cosine similarity with normalisation
    v1 = embed1.detach().cpu().numpy()
    v2 = embed2.detach().cpu().numpy()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    v1_normalised = v1 / norm1
    v2_normalised = v2 / norm2

    return np.dot(v1_normalised, v2_normalised.T)

def compute_sim_matrix(eng_word_emb, hin_word_emb, eng_sent_emb, hin_sent_emb, threshold=0.5):
    sent_sim = similarity(eng_sent_emb, hin_sent_emb)
    if sent_sim < threshold:
        return None
    
    # Shape: [1, seq_len, 768] -> extracting dimensions
    num_eng_tokens = eng_word_emb.shape[1]
    num_hin_tokens = hin_word_emb.shape[1]
    
    # Initialising matrix
    matrix = np.zeros((num_eng_tokens, num_hin_tokens))
    
    for j in range(num_eng_tokens):
        for k in range(num_hin_tokens):
            # Slice correctly: [0, j, :] gives shape [768]
            eng_token_vec = eng_word_emb[0, j, :]
            hin_token_vec = hin_word_emb[0, k, :]
            matrix[j, k] = similarity(eng_token_vec, hin_token_vec)
    
    return matrix

def per_row_argmax(sim_matrix):
    alignments = []
    for i in range(sim_matrix.shape[0]):
        best_match = np.argmax(sim_matrix[i])
        alignments.append((i, int(best_match)))
    return alignments

def bidir_argmax(sim_matrix):
    forward = []
    for i in range(sim_matrix.shape[0]):
        best_match = np.argmax(sim_matrix[i])
        forward.append((i, int(best_match)))

    backward = []
    for j in range(sim_matrix.shape[1]):
        best_match = np.argmax(sim_matrix[:, j])
        backward.append((int(best_match), j))

    # Set intersection works better for tuples
    final_alignment = set(forward) & set(backward)

    return list(final_alignment)

def convert_id_to_token(alignments, eng_sentence, hin_sentence):
    # Get tokens with special markers
    eng_tokens = ["[CLS]"] + tokenizer.tokenize(eng_sentence) + ["[SEP]"]
    hin_tokens = ["[CLS]"] + tokenizer.tokenize(hin_sentence) + ["[SEP]"]
    
    aligned_pairs = []
    
    for (eng_idx, hin_idx) in alignments:
        # Skip special tokens [CLS] (index 0) and [SEP] (last index)
        if eng_idx == 0 or eng_idx == len(eng_tokens) - 1:
            continue
        if hin_idx == 0 or hin_idx == len(hin_tokens) - 1:
            continue
        
        eng_token = eng_tokens[eng_idx]
        hin_token = hin_tokens[hin_idx]
        
        aligned_pairs.append((eng_token, hin_token))
    
    return aligned_pairs

def merge_subwords(aligned_pairs):
    merged = []
    current_eng = ""
    current_hin = ""
    
    for (eng_tok, hin_tok) in aligned_pairs:
        if eng_tok.startswith("##"):
            current_eng += eng_tok[2:]  # Remove ## prefix
        else:
            if current_eng:
                merged.append((current_eng, current_hin))
            current_eng = eng_tok
            current_hin = hin_tok
    
    # Don't forget the last pair
    if current_eng:
        merged.append((current_eng, current_hin))
    
    return merged

def build_dictionary(all_alignments):
    from collections import defaultdict
    
    dictionary = defaultdict(int)
    
    for (eng, hin) in all_alignments:
        # Clean tokens
        eng_clean = eng.lower().strip(string.punctuation)
        hin_clean = hin.strip()
        
        if eng_clean and hin_clean:  # Skip empty tokens
            dictionary[(eng_clean, hin_clean)] += 1
    
    return dict(dictionary)

def save_dictionary(dictionary, output_path="output/bilingual_dict.txt"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for (eng, hin), count in sorted_dict:
            f.write(f"{eng}\t{hin}\t{count}\n")
    
    print(f"Dictionary saved to {output_path} with {len(sorted_dict)} entries")


if __name__=="__main__":
    NUM_SENTENCES = 100
    
    print("Step 1: Getting embeddings...")
    english_embeddings, hindi_embeddings = get_sentence_embeddings(NUM_SENTENCES)
    english_word_embeddings, hindi_word_embeddings = get_word_embeddings(NUM_SENTENCES)
    
    print("\nStep 2: Computing alignments...")
    all_alignments = []
    
    for i in range(NUM_SENTENCES):
        matrix = compute_sim_matrix(
            english_word_embeddings[i],
            hindi_word_embeddings[i],
            english_embeddings[i],
            hindi_embeddings[i]
        )
        
        if matrix is not None:
            # Get bidirectional alignments
            alignments = bidir_argmax(matrix)
            
            # Convert indices to tokens
            token_pairs = convert_id_to_token(
                alignments,
                english_sentences[i],
                hindi_sentences[i]
            )
            
            # Merge subwords
            merged_pairs = merge_subwords(token_pairs)
            
            # Add to all alignments
            all_alignments.extend(merged_pairs)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{NUM_SENTENCES} sentences")
    
    print(f"\nStep 3: Building dictionary from {len(all_alignments)} alignment pairs...")
    dictionary = build_dictionary(all_alignments)
    
    print(f"\nStep 4: Saving dictionary...")
    save_dictionary(dictionary)