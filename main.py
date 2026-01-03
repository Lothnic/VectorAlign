import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer

# CONFIG
MODEL_NAME = "setu4993/LaBSE"

device = "cuda" if torch.cuda_is_available() else "cpu"
print(f"Using device : {device}")

# Loading the LaBSE model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


