"""
Unified Training Module

Provides training functions for both DDL and ReFT approaches,
using real embeddings from the Constitutional dataset.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from src.llm_client import get_llm_client
from src.layers.delta import MultiHeadDelta, MultiHeadLoReFT
from src.dataset.personas import SampleDB, RunManager


@dataclass
class TrainingResult:
    """Results from a training run."""
    model_path: Path
    method: str  # 'ddl' or 'reft'
    embed_dim: int
    num_samples: int
    epochs: int
    initial_loss: float
    final_loss: float
    improvement_pct: float


class EmbeddingCache:
    """Cache for text embeddings to avoid redundant API calls."""
    
    def __init__(self, client):
        self.client = client
        self._cache = {}
        self.embed_dim = None
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get embedding from cache or compute it."""
        text_key = hash(text[:1000])
        
        if text_key in self._cache:
            return self._cache[text_key]
        
        try:
            emb = self.client.embed(text[:8000])
            if emb:
                tensor = torch.tensor(emb, dtype=torch.float32)
                self._cache[text_key] = tensor
                if self.embed_dim is None:
                    self.embed_dim = len(emb)
                return tensor
        except Exception as e:
            print(f"[WARN] Embedding failed: {e}")
        
        return None
    
    def precompute_samples(self, samples: list, progress_fn: Callable = None) -> tuple[list, list]:
        """
        Precompute embeddings for all samples.
        
        Returns:
            (bad_embeddings, good_embeddings) - lists of tensors
        """
        import time
        bad_embs = []
        good_embs = []
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            elapsed = time.time() - start_time
            if i > 0:
                rate = i / elapsed
                remaining = (len(samples) - i) / rate
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f" | ETA: {eta_min}m {eta_sec}s" if eta_min > 0 else f" | ETA: {eta_sec}s"
            else:
                eta_str = ""
            
            if progress_fn:
                progress_fn(i / len(samples), f"Embedding {i+1}/{len(samples)}{eta_str}")
            
            naive = sample.get("naive_response", "")
            revised = sample.get("revised_response", "")
            
            emb_bad = self.get(naive)
            emb_good = self.get(revised)
            
            if emb_bad is not None and emb_good is not None:
                bad_embs.append(emb_bad)
                good_embs.append(emb_good)
        
        return bad_embs, good_embs


def train_ddl(
    samples: list,
    client,
    epochs: int = 5,
    learning_rate: float = 0.001,
    num_heads: int = 4,
    progress_fn: Callable = None
) -> TrainingResult:
    """
    Train Deep Delta Learning model.
    
    Learns to transform "bad" (naive) embeddings toward "good" (revised) embeddings.
    
    Args:
        samples: List of sample dicts with 'naive_response' and 'revised_response'
        client: LLMClient for embeddings
        epochs: Number of training epochs
        learning_rate: Learning rate
        num_heads: Number of delta heads
        progress_fn: Optional callback fn(progress_pct, message)
    
    Returns:
        TrainingResult with model path and metrics
    """
    torch.set_default_device('cpu')
    
    # Precompute embeddings
    cache = EmbeddingCache(client)
    if progress_fn:
        progress_fn(0.0, "Generating embeddings...")
    
    bad_embs, good_embs = cache.precompute_samples(
        samples, 
        lambda p, m: progress_fn(p * 0.4, m) if progress_fn else None
    )
    
    if not bad_embs:
        raise ValueError("No embeddings generated. Check embedding model configuration.")
    
    embed_dim = cache.embed_dim
    X_bad = torch.stack(bad_embs)
    X_good = torch.stack(good_embs)
    
    if progress_fn:
        progress_fn(0.45, f"Training on {len(X_bad)} samples (dim={embed_dim})...")
    
    # Create and train model
    model = MultiHeadDelta(embed_dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(len(X_bad))
        
        for i in range(len(X_bad)):
            x_bad = X_bad[perm[i]:perm[i]+1]
            x_good = X_good[perm[i]:perm[i]+1]
            
            optimizer.zero_grad()
            output = model(x_bad)
            loss = (output - x_good).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(X_bad)
        losses.append(avg_loss)
        
        if progress_fn:
            progress_fn(0.45 + 0.5 * ((epoch + 1) / epochs), 
                       f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Save model
    run_dir = RunManager.get_run_dir()
    save_path = run_dir / "ddl_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "training_samples": len(X_bad),
        "losses": losses
    }, save_path)
    
    if progress_fn:
        progress_fn(1.0, "Done!")
    
    return TrainingResult(
        model_path=save_path,
        method="ddl",
        embed_dim=embed_dim,
        num_samples=len(X_bad),
        epochs=epochs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        improvement_pct=((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0
    )


def train_reft(
    samples: list,
    client,
    epochs: int = 5,
    learning_rate: float = 0.01,
    rank: int = 4,
    num_heads: int = 4,
    progress_fn: Callable = None
) -> TrainingResult:
    """
    Train ReFT (Representation Fine-Tuning) model.
    
    Uses MultiHeadLoReFT to learn low-rank interventions that steer
    embeddings from naive toward revised responses.
    
    Args:
        samples: List of sample dicts with 'naive_response' and 'revised_response'
        client: LLMClient for embeddings
        epochs: Number of training epochs
        learning_rate: Learning rate
        rank: Intervention rank per head
        num_heads: Number of LoReFT heads
        progress_fn: Optional callback fn(progress_pct, message)
    
    Returns:
        TrainingResult with model path and metrics
    """
    torch.set_default_device('cpu')
    
    # Precompute embeddings
    cache = EmbeddingCache(client)
    if progress_fn:
        progress_fn(0.0, "Generating embeddings...")
    
    bad_embs, good_embs = cache.precompute_samples(
        samples,
        lambda p, m: progress_fn(p * 0.4, m) if progress_fn else None
    )
    
    if not bad_embs:
        raise ValueError("No embeddings generated. Check embedding model configuration.")
    
    embed_dim = cache.embed_dim
    X_bad = torch.stack(bad_embs)
    X_good = torch.stack(good_embs)
    
    if progress_fn:
        progress_fn(0.45, f"Training ReFT on {len(X_bad)} samples (dim={embed_dim}, rank={rank})...")
    
    # Create LoReFT model
    model = MultiHeadLoReFT(embed_dim, num_heads=num_heads, rank=rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(len(X_bad))
        
        for i in range(len(X_bad)):
            x_bad = X_bad[perm[i]:perm[i]+1]
            x_good = X_good[perm[i]:perm[i]+1]
            
            optimizer.zero_grad()
            
            # LoReFT: apply intervention to bad embedding
            output = model(x_bad)
            
            # Loss: intervened output should match good embedding
            loss = (output - x_good).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(X_bad)
        losses.append(avg_loss)
        
        if progress_fn:
            progress_fn(0.45 + 0.5 * ((epoch + 1) / epochs),
                       f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    # Save model
    run_dir = RunManager.get_run_dir()
    save_path = run_dir / "reft_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "rank": rank,
        "training_samples": len(X_bad),
        "losses": losses
    }, save_path)
    
    if progress_fn:
        progress_fn(1.0, "Done!")
    
    return TrainingResult(
        model_path=save_path,
        method="reft",
        embed_dim=embed_dim,
        num_samples=len(X_bad),
        epochs=epochs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        improvement_pct=((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] > 0 else 0
    )


def load_trained_model(run_id: str, method: str = "ddl") -> tuple[nn.Module, dict]:
    """
    Load a trained model from a run.
    
    Args:
        run_id: Run ID to load from
        method: 'ddl' or 'reft'
    
    Returns:
        (model, metadata) tuple
    """
    RunManager.set_run(run_id)
    run_dir = RunManager.get_run_dir()
    
    filename = f"{method}_model.pt"
    model_path = run_dir / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"No {method} model found at {model_path}")
    
    checkpoint = torch.load(model_path, weights_only=False)
    
    embed_dim = checkpoint["embed_dim"]
    num_heads = checkpoint["num_heads"]
    
    if method == "ddl":
        model = MultiHeadDelta(embed_dim, num_heads=num_heads)
    else:
        rank = checkpoint.get("rank", 4)
        model = MultiHeadLoReFT(embed_dim, num_heads=num_heads, rank=rank)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, checkpoint
