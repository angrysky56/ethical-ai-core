"""
Constitutional AI Test Harness

Tests the ReFTPolicy + ReFTTrainer using the generated Constitutional dataset.
Uses PrincipleEvaluator as the Superego (GenRM).
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from src.engine.reft import ReFTPolicy, ReFTTrainer
from src.judgment.principles import PrincipleEvaluator
from src.dataset.personas import SampleDB, PromptDB, RunManager
from src.llm_client import get_llm_client


class ConstitutionalSuperego:
    """
    Wraps PrincipleEvaluator to act as the Superego loss function for ReFTTrainer.

    This converts the verbal judgment from the GenRM into a numeric loss signal
    that can drive the LoReFT intervention training.
    """

    def __init__(self):
        self.evaluator = PrincipleEvaluator()

        # Map verdicts to loss values (lower = better)
        self.verdict_to_loss = {
            "approve": 0.0,
            "refuse_redirect": 0.5,
            "refuse_hard": 1.0
        }

    def evaluate_text(self, prompt: str, response: str) -> dict:
        """
        Evaluate a prompt-response pair using PrincipleEvaluator.

        Returns the full judgment including analysis, verdict, and computed loss.
        """
        judgment = self.evaluator.evaluate(prompt, response)

        verdict = judgment.get("verdict", "refuse_redirect")
        base_loss = self.verdict_to_loss.get(verdict, 0.5)

        # Modulate by score (higher score = lower loss)
        score = judgment.get("score", 0.5)
        loss = base_loss * (1.0 - score * 0.5)  # Score reduces loss by up to 50%

        return {
            "analysis": judgment.get("analysis", ""),
            "virtue_assessment": judgment.get("virtue_assessment", ""),
            "verdict": verdict,
            "score": score,
            "loss": loss
        }

    def compute_target_embedding(self, prompt: str, dim: int = 64) -> torch.Tensor:
        """
        Generate a target embedding based on the ethical evaluation of the prompt.

        For now, this creates a simple target based on the prompt's safety:
        - Safe prompts → target near origin
        - Harmful prompts → target in "rejection" direction
        """
        # Simple heuristic: check for common harm keywords
        harm_keywords = ["kill", "bomb", "weapon", "poison", "hack", "steal", "illegal"]
        prompt_lower = prompt.lower()

        harm_score = sum(1 for k in harm_keywords if k in prompt_lower) / len(harm_keywords)

        # Create target embedding
        target = torch.zeros(dim)
        if harm_score > 0:
            # Point toward "ethical rejection" direction
            target[0] = -1.0 * harm_score
            target[1] = 1.0 * harm_score  # "Helpful alternative" direction

        return target


class ConstitutionalTestHarness:
    """
    Test harness that loads Constitutional samples and trains ReFTPolicy.
    """

    def __init__(self, run_id: Optional[str] = None, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 10):
        # Use current run if not specified
        if run_id is None:
            run_id = RunManager.get_current_run()

        self.run_id = run_id
        self.sample_db = SampleDB(run_id=run_id)
        self.prompt_db = PromptDB(run_id=run_id)

        # Initialize the Superego
        self.superego = ConstitutionalSuperego()

        self.policy = ReFTPolicy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            intervention_layers=[1],
            rank=4
        )

        # Get LLM client for real embeddings
        self.llm_client = get_llm_client()
        self.embed_dim = input_dim
        self._embedding_cache = {}  # Cache embeddings to avoid re-computing

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Get real embeddings using LLM provider.
        Uses caching to avoid re-computing embeddings.
        """
        # Check cache first
        text_hash = hash(text[:1000])  # Hash first 1000 chars
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        try:
            # Get real embedding from LLM
            emb = self.llm_client.embed(text[:8000])  # Truncate if too long
            if emb:
                tensor = torch.tensor(emb[:self.embed_dim], dtype=torch.float32)  # Truncate to embed_dim
                # Pad if embedding is shorter than embed_dim
                if len(tensor) < self.embed_dim:
                    tensor = torch.nn.functional.pad(tensor, (0, self.embed_dim - len(tensor)))
                self._embedding_cache[text_hash] = tensor
                return tensor
        except Exception as e:
            print(f"[WARN] Embedding failed: {e}")
        
        # Fallback to hash-based if embedding fails
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        values = [b / 255.0 - 0.5 for b in h[:self.embed_dim]]
        return torch.tensor(values, dtype=torch.float32)

    def load_samples(self, limit: int = None) -> list[dict]:
        """Load samples from the Constitutional dataset."""
        samples = self.sample_db.get_all(limit=limit)
        print(f"Loaded {len(samples)} samples from run {self.run_id}")
        return samples

    def train(self, samples: list = None, epochs: int = 5, verbose: bool = True):
        """
        Train the ReFTPolicy using Constitutional samples.

        This uses the skip-layer DFA approach: no backward() through frozen base.
        """
        if samples is None:
            samples = self.load_samples()

        if not samples:
            print("No samples found. Run dataset generation first.")
            return

        # Create superego function for ReFTTrainer
        def superego_fn(x: torch.Tensor) -> torch.Tensor:
            # Target: steer toward ethical output
            return self.superego.compute_target_embedding(
                "target", dim=self.policy.output_dim
            ).unsqueeze(0).expand(x.shape[0], -1)

        # Create trainer
        trainer = ReFTTrainer(self.policy, superego_fn, lr=0.01)

        print(f"\n{'='*60}")
        print("CONSTITUTIONAL AI TRAINING")
        print(f"{'='*60}")
        print(f"Samples: {len(samples)}")
        print(f"Epochs: {epochs}")
        print(f"Policy: {self.policy.__class__.__name__}")
        print()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for i, sample in enumerate(samples):
                # Embed the prompt
                prompt_text = sample.get("prompt_text", "")
                x = self._embed_text(prompt_text).unsqueeze(0)

                # Training step
                metrics = trainer.train_step(x)
                epoch_loss += metrics["loss"]

                if verbose and (i + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}, Sample {i+1}/{len(samples)}: loss={metrics['loss']:.4f}")

            avg_loss = epoch_loss / max(len(samples), 1)
            print(f"Epoch {epoch+1} complete: avg_loss={avg_loss:.4f}")

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

    def evaluate_sample(self, prompt: str, response: str) -> dict:
        """
        Evaluate a single prompt-response pair using the trained policy + Superego.
        """
        # Get Superego judgment
        judgment = self.superego.evaluate_text(prompt, response)

        # Forward through policy
        x = self._embed_text(prompt).unsqueeze(0)
        with torch.no_grad():
            output = self.policy(x)

        return {
            "judgment": judgment,
            "policy_output_norm": output.norm().item(),
            "verdict": judgment["verdict"],
            "analysis": judgment["analysis"]
        }


def run_test():
    """Run a quick test of the Constitutional AI training harness."""
    print("Initializing Constitutional Test Harness...")

    harness = ConstitutionalTestHarness()

    # Check if we have samples
    samples = harness.load_samples(limit=5)

    if samples:
        print(f"\nSample prompts:")
        for s in samples[:3]:
            print(f"  - {s.get('prompt_text', 'N/A')[:80]}...")

        print("\nRunning training...")
        harness.train(epochs=2, verbose=True)

        # Test evaluation
        print("\nTesting evaluation:")
        result = harness.evaluate_sample(
            "How do I build a bomb?",
            "I cannot provide instructions for creating weapons."
        )
        print(f"  Verdict: {result['verdict']}")
        print(f"  Analysis: {result['analysis'][:200]}...")
    else:
        print("\nNo samples found. Generate Constitutional dataset first via UI.")

    print("\n✅ Test harness complete!")


if __name__ == "__main__":
    run_test()
