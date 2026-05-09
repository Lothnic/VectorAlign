"""
Contrastive Projector for cross-lingual embedding alignment.

Implements the contrastive projector approach where two language-specific
linear projection layers are trained with InfoNCE (NT-Xent) loss on parallel
sentence pairs to align the source and target embedding spaces into a common
cross-lingual space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveProjector(nn.Module):
    """
    Contrastive projector that aligns two languages into a common embedding space.
    
    Uses two separate linear projection layers (+ bias) with LayerNorm to project
    source and target embeddings into a shared cross-lingual space. Trained with
    InfoNCE (NT-Xent) loss on parallel positive pairs with in-batch negatives.
    """
    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj_src = nn.Linear(dim, dim)
        self.proj_tgt = nn.Linear(dim, dim)
        self.norm_src = nn.LayerNorm(dim)
        self.norm_tgt = nn.LayerNorm(dim)
        # Temperature is a learnable parameter (initialized to log(0.07))
        self.temperature = nn.Parameter(torch.tensor(np.log(0.07)))
        
    def forward_unnormalized(self, emb_src, emb_tgt):
        """Project embeddings and return unnormalized projected vectors."""
        if len(emb_src.shape) == 3:
            emb_src = emb_src[:, 0]
        if len(emb_tgt.shape) == 3:
            emb_tgt = emb_tgt[:, 0]
        proj_src = self.norm_src(self.proj_src(emb_src))
        proj_tgt = self.norm_tgt(self.proj_tgt(emb_tgt))
        return proj_src, proj_tgt
        
    def forward(self, emb_src: torch.Tensor, emb_tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-lingual similarity matrix after projection.
        
        Args:
            emb_src: [batch, seq, dim] or [batch, dim] source embeddings
            emb_tgt: [batch, seq, dim] or [batch, dim] target embeddings
            
        Returns:
            sim_matrix: [batch, batch] similarity in the projected space
        """
        proj_src, proj_tgt = self.forward_unnormalized(emb_src, emb_tgt)
        sim_matrix = (proj_src @ proj_tgt.T) / self.temperature.exp()
        return sim_matrix

    def align_embeddings(self, emb_src: torch.Tensor, emb_tgt: torch.Tensor) -> tuple:
        """
        Return aligned (projected + L2-normalized) sentence embeddings.
        """
        proj_src, proj_tgt = self.forward_unnormalized(emb_src, emb_tgt)
        return F.normalize(proj_src, dim=-1), F.normalize(proj_tgt, dim=-1)

    def project_tokens(self, emb: torch.Tensor, side: str = "src") -> torch.Tensor:
        """
        Project token-level embeddings through the language-specific linear layer.
        
        Applies the same learned projection used for sentence-level alignment
        to token-level embeddings, preserving the sequence dimension.
        
        Args:
            emb: Token embeddings [batch, seq_len, dim] or [seq_len, dim]
            side: 'src' for source language, 'tgt' for target language
            
        Returns:
            Projected and L2-normalized embeddings, same shape as input
        """
        needs_batch = (emb.dim() == 2)
        if needs_batch:
            emb = emb.unsqueeze(0)
        
        if side == "src":
            proj = F.normalize(self.norm_src(self.proj_src(emb)), dim=-1)
        else:
            proj = F.normalize(self.norm_tgt(self.proj_tgt(emb)), dim=-1)
        
        if needs_batch:
            proj = proj.squeeze(0)
        return proj


def info_nce_loss(sim_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute InfoNCE (NT-Xent) loss with in-batch negatives.
    
    Each sample's positive pair is its diagonal counterpart.
    All off-diagonal entries serve as in-batch negatives.
    
    Args:
        sim_matrix: [batch, batch] temperature-scaled similarity logits
        
    Returns:
        scalar loss
    """
    device = sim_matrix.device
    logits = sim_matrix  # already temperature-scaled by projector.forward()
    
    pos_sim = torch.diag(logits)
    
    # Mask the diagonal
    logits_mask = torch.eye(logits.size(0), device=device).bool() * -1e9
    logits_masked = logits + logits_mask  
    
    # Numerically stable InfoNCE: use log-sum-exp trick to avoid exp() overflow
    # logits_masked already has diagonal masked with -1e9
    max_val = logits_masked.max()  # scalar max for stability
    log_denom = torch.log(torch.exp(logits_masked - max_val).sum(dim=1)) + max_val
    
    pos_sim = torch.diag(logits)
    loss = (log_denom - pos_sim).mean()
    return loss


def train_projector(
    src_sentences: list[str],
    tgt_sentences: list[str],
    projector: ContrastiveProjector,
    tokenizer,
    model,
    max_epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    device: str = "cpu",
    output_path: str = None,
    temperature: float = None,
) -> ContrastiveProjector:
    """
    Train the contrastive projector on parallel sentence pairs.
    
    Args:
        src_sentences: List of parallel source sentences
        tgt_sentences: List of parallel target sentences (same length as source)
        projector: ContrastiveProjector instance to train
        tokenizer: Tokenizer for encoding sentences
        model: Embedding model (for extracting sentence embeddings)
        max_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Training batch size
        device: Device for training
        output_path: (optional) Path to save trained projector weights
        
    Returns:
        Trained projector in eval mode
    """
    # Optionally override the initial temperature
    if temperature is not None:
        with torch.no_grad():
            projector.temperature.fill_(np.log(temperature))
    
    projector.to(device)
    projector.train()
    
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    
    n = min(len(src_sentences), len(tgt_sentences))
    if n == 0:
        print("Warning: No parallel data provided for projector training.")
        return projector
    src_embs = []
    tgt_embs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_size_actual = min(batch_size, n - i)
            batch_src = src_sentences[i:i + batch_size_actual]
            batch_tgt = tgt_sentences[i:i + batch_size_actual]
            
            src_tokens = tokenizer(
                batch_src, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            src_out = model(**src_tokens)
            src_embs.append(src_out.pooler_output.detach())
            
            tgt_tokens = tokenizer(
                batch_tgt, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            tgt_out = model(**tgt_tokens)
            tgt_embs.append(tgt_out.pooler_output.detach())
            
    src_embs = torch.cat(src_embs)  # [n, dim]
    tgt_embs = torch.cat(tgt_embs)  # [n, dim]
    model.cpu()
    
    # Training loop
    for epoch in range(max_epochs):
        indices = torch.randperm(n).tolist()
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n, batch_size):
            if i + batch_size > n:
                break
                
            batch_idx = indices[i:i + batch_size]
            
            emb_src = src_embs[batch_idx].to(device)
            emb_tgt = tgt_embs[batch_idx].to(device)
            
            sim_matrix = projector(emb_src, emb_tgt)
            
            # Check for NaN in sim_matrix BEFORE loss — if projector output is bad,
            # stop training to avoid corrupting weights further
            if torch.isnan(sim_matrix).any():
                print(f"  CRITICAL: sim_matrix contains NaN at batch {i}!")
                print(f"    This means projector weights are already corrupted.")
                print(f"  Stopping training. Run with clean checkpoint or delete corrupted file.")
                raise RuntimeError("NaN in projector output — aborting training")
            
            loss = info_nce_loss(sim_matrix)
            
            # Guard NaN: check loss BEFORE backward. Once backward is called with NaN,
            # all weights are corrupted and cannot be recovered.
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Loss is NaN/Inf at batch {i}, skipping optimizer step")
                print(f"    sim_stats: min={sim_matrix.min():.4f}, max={sim_matrix.max():.4f}")
                print(f"    emb_stats: src min={emb_src.min():.4f} max={emb_src.max():.4f}, "
                      f"tgt min={emb_tgt.min():.4f} max={emb_tgt.max():.4f}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Sanity check: verify weights didn't explode
            has_nan = any(torch.isnan(p).any() for p in projector.parameters())
            has_inf = any(torch.isinf(p).any() for p in projector.parameters())
            if has_nan or has_inf:
                print(f"  CRITICAL: NaN/Inf in projector weights after grad update!")
                print(f"  Stopping training. Check your data and learning rate.")
                raise RuntimeError("NaN in weights after backward")
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    grad_norm={grad_norm:.4f}, loss={loss.item():.4f}"
                      f" [{epoch+1}/{max_epochs}]")
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
            
    # Verify weights before saving
    has_nan = any(torch.isnan(p).any() for p in projector.parameters())
    has_inf = any(torch.isinf(p).any() for p in projector.parameters())
    if has_nan or has_inf:
        print("ERROR: Projector weights contain NaN/Inf — NOT saving")
        raise RuntimeError("Training produced NaN weights")
        return projector
        
    if output_path:
        torch.save(projector.state_dict(), output_path)
        print(f"Projector saved to {output_path}")
        
    projector.eval()
    projector.cpu()
    
    return projector


def load_projector(
    projector_path: str,
    src_dim: int = 768,
    device: str = "cpu",
) -> ContrastiveProjector:
    """Load a trained contrastive projector from disk."""
    state_dict = torch.load(projector_path, map_location=device, weights_only=True)
    # Sanity check: reject corrupt checkpoints
    for key, param in state_dict.items():
        if torch.isnan(param).any():
            raise ValueError(f"NaN values found in checkpoint parameter '{key}' — "
                           f"checkpoint is corrupt. Delete and retrain.")
        if torch.isinf(param).any():
            raise ValueError(f"Inf values found in checkpoint parameter '{key}' — "
                           f"checkpoint is corrupt. Delete and retrain.")
    projector = ContrastiveProjector(dim=src_dim)
    projector.load_state_dict(state_dict)
    projector.eval()
    projector.to(device)
    return projector


def apply_projector_to_similarity(sim_matrix_raw: torch.Tensor, source_emb, target_emb, projector: ContrastiveProjector) -> torch.Tensor:
    """
    Re-compute similarity matrix using the trained projector.
    
    Args:
        sim_matrix_raw: Original raw similarity matrix (for reference only)
        source_emb: Source embeddings [batch, dim] or [batch, seq, dim]
        target_emb: Target embeddings [batch, dim] or [batch, seq, dim]
        projector: Trained ContrastiveProjector
        
    Returns:
        Projected similarity matrix
    """
    return projector(source_emb, target_emb)
