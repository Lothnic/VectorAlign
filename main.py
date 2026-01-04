import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
import numpy as np
from tqdm import tqdm
import string

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

def get_embeddings_batch(src_sentences: list[str], tgt_sentences: list[str],tokenizer,model,batch_size=32,device='cpu'):
    src_embeddings = []
    tgt_embeddings = []
    src_word_embeddings = []
    tgt_word_embeddings = []
    
    length = min(len(src_sentences),len(tgt_sentences))
    with torch.no_grad():
        for i in range(0,length,batch_size):
            
            batch_src = src_sentences[i:i+batch_size]
            batch_tgt = tgt_sentences[i:i+batch_size]

            src_tokens = tokenizer(batch_src, return_tensors='pt', padding=True, truncation=True).to(device)
            tgt_tokens = tokenizer(batch_tgt, return_tensors='pt', padding=True, truncation=True).to(device)

            '''
            Up we were tokenising entire batch at once

            Now we are doing the single forward pass for the batch of sentences
            '''
            src_out = model(**src_tokens)
            tgt_out = model(**tgt_tokens)

            src_embeddings.append(src_out.pooler_output.detach().cpu())
            tgt_embeddings.append(tgt_out.pooler_output.detach().cpu())
            src_word_embeddings.append(src_out.last_hidden_state.detach().cpu())
            tgt_word_embeddings.append(tgt_out.last_hidden_state.detach().cpu())

    src_embeddings = torch.cat(src_embeddings, dim=0)
    tgt_embeddings = torch.cat(tgt_embeddings, dim=0)

    src_embeddings = [e.unsqueeze(0) for e in src_embeddings]
    tgt_embeddings = [e.unsqueeze(0) for e in tgt_embeddings]

    # Flatten word embeddings: each batch has different seq_len, so we split individually
    src_word_flat = []
    tgt_word_flat = []
    
    for batch in src_word_embeddings:
        for j in range(batch.shape[0]):
            src_word_flat.append(batch[j:j+1])  # [1, seq_len, 768]

    for batch in tgt_word_embeddings:
        for j in range(batch.shape[0]):
            tgt_word_flat.append(batch[j:j+1])  # [1, seq_len, 768]

    return src_embeddings, tgt_embeddings, src_word_flat, tgt_word_flat


def similarity(embed1,embed2):
    # return cosine similarity with normalisation
    v1 = embed1.detach().cpu().numpy()
    v2 = embed2.detach().cpu().numpy()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    v1_normalised = v1 / norm1
    v2_normalised = v2 / norm2

    return np.dot(v1_normalised, v2_normalised.T)

def compute_sim_matrix(src_word_emb, tgt_word_emb, src_sent_emb, tgt_sent_emb, threshold=0.5):
    sent_sim = similarity(src_sent_emb, tgt_sent_emb)
    if sent_sim < threshold:
        return None
    
    # Shape: [1, seq_len, 768] -> extracting dimensions
    num_src_tokens = src_word_emb.shape[1]
    num_tgt_tokens = tgt_word_emb.shape[1]
    
    # Initialising matrix
    matrix = np.zeros((num_src_tokens, num_tgt_tokens))
    
    for j in range(num_src_tokens):
        for k in range(num_tgt_tokens):
            src_token_vec = src_word_emb[0, j, :]
            tgt_token_vec = tgt_word_emb[0, k, :]
            matrix[j, k] = similarity(src_token_vec, tgt_token_vec)
    
    return matrix

def compute_sim_matrix_batch(src_word_emb, tgt_word_emb, src_sent_emb, tgt_sent_emb, threshold=0.5):
    '''
    Computation of matrix operation is faster in pytorch due to C optimisations.
    '''
    sent_sim = F.cosine_similarity(src_sent_emb, tgt_sent_emb, dim=-1).item()
    
    if sent_sim < threshold:
        return None

    # Squeeze to [seq_len, hidden_size]
    src_tokens = src_word_emb.squeeze(0)  # [src_len, 768]
    tgt_tokens = tgt_word_emb.squeeze(0)  # [tgt_len, 768]
    
    src_norm = F.normalize(src_tokens, dim=-1)
    tgt_norm = F.normalize(tgt_tokens, dim=-1)
    
    # MatMul gives cosine similarity when vectors are normalized
    sim_matrix = torch.matmul(src_norm, tgt_norm.T) 
    return sim_matrix

def per_row_argmax(sim_matrix):
    alignments = []
    for i in range(sim_matrix.shape[0]):
        best_match = np.argmax(sim_matrix[i])
        alignments.append((i, int(best_match)))
    return alignments

def bidir_argmax(sim_matrix):
    forward = []
    for i in range(sim_matrix.shape[0]):
        best_match = torch.argmax(sim_matrix[i])
        forward.append((i, int(best_match)))

    backward = []
    for j in range(sim_matrix.shape[1]):
        best_match = torch.argmax(sim_matrix[:, j])
        backward.append((int(best_match), j))

    final_alignment = set(forward) & set(backward)

    return list(final_alignment)

def convert_id_to_token(alignments, src_sentence, tgt_sentence, tokenizer):
    # Get tokens with special markers
    src_tokens = ["[CLS]"] + tokenizer.tokenize(src_sentence) + ["[SEP]"]
    tgt_tokens = ["[CLS]"] + tokenizer.tokenize(tgt_sentence) + ["[SEP]"]
    
    aligned_pairs = []
    
    for (src_idx, tgt_idx) in alignments:
        # Skip if index is out of bounds (PAD token positions from batching)
        if src_idx >= len(src_tokens) or tgt_idx >= len(tgt_tokens):
            continue
        
        # Skip special tokens [CLS] (index 0) and [SEP] (last index)
        if src_idx == 0 or src_idx == len(src_tokens) - 1:
            continue
        if tgt_idx == 0 or tgt_idx == len(tgt_tokens) - 1:
            continue
        
        src_token = src_tokens[src_idx]
        tgt_token = tgt_tokens[tgt_idx]
        
        aligned_pairs.append((src_token, tgt_token))
    
    return aligned_pairs

def merge_subwords(aligned_pairs):
    merged = []
    current_src = ""
    current_tgt = ""
    
    for (src_tok, tgt_tok) in aligned_pairs:
        if src_tok.startswith("##"):
            current_src += src_tok[2:]  # Remove ## prefix
        else:
            if current_src:
                merged.append((current_src, current_tgt))
            current_src = src_tok
            current_tgt = tgt_tok
    
    # Don't forget the last pair
    if current_src:
        merged.append((current_src, current_tgt))
    
    return merged

def build_dictionary(all_alignments):
    from collections import defaultdict
    
    dictionary = defaultdict(int)
    
    for (src, tgt) in all_alignments:
        src_clean = src.lower().strip(string.punctuation)
        tgt_clean = tgt.strip()
        
        if src_clean and tgt_clean:
            dictionary[(src_clean, tgt_clean)] += 1
    
    return dict(dictionary)

def save_dictionary(dictionary, output_path="output/dict.txt"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for (src, tgt), count in sorted_dict:
            if count > 1:
                f.write(f"{src}\t{tgt}\t{count}\n")
    
    print(f"Dictionary saved to {output_path}")

def align(src_sentences: list[str], tgt_sentences: list[str],batch_size: int=32,model_name="bert",mode='multilingual',output='output/dict.txt'):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device : {device}")

    if model_name=="bert":
        if mode=='multilingual':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)
            model.eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased").to(device)
            model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Model Loaded")
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

    NUM_SENTENCES = len(src_sentences)
    src_embeddings, tgt_embeddings, src_word_embeddings, tgt_word_embeddings = get_embeddings_batch(src_sentences, tgt_sentences, tokenizer, model, batch_size,device)
    
    all_alignments = []
    
    for i in tqdm(range(NUM_SENTENCES),desc="Aligning sentences"):
        matrix = compute_sim_matrix_batch(
            src_word_embeddings[i],
            tgt_word_embeddings[i],
            src_embeddings[i],
            tgt_embeddings[i]
        )
        
        if matrix is not None:
            alignments = bidir_argmax(matrix)

            token_pairs = convert_id_to_token(
                alignments,
                src_sentences[i],
                tgt_sentences[i],
                tokenizer
            )
            merged_pairs = merge_subwords(token_pairs)
            all_alignments.extend(merged_pairs)

    dictionary = build_dictionary(all_alignments)
    save_dictionary(dictionary,output)

    
if __name__=="__main__":
        
    # CONFIG
    MODEL_NAME = "setu4993/LaBSE"

    # Loading the data
    with open('data/english.txt','r', encoding='utf-8') as f:
        english_sentences = f.read().strip().strip('\ufeff').split('\n')

    with open('data/hindi.txt','r',encoding='utf-8') as f:
        hindi_sentences = f.read().strip().strip('\ufeff').split('\n')

    align(english_sentences,hindi_sentences,model_name=MODEL_NAME,batch_size=128)
