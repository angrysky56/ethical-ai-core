"""
Constitutional AI Dataset Generator

Implements Section 3.1 of Bootstrapping-Core-Self-via-AI-Feedback.md:
1. Red Teaming: Generate adversarial/dilemma prompts
2. Naive Response: Get base model response
3. Critique: Teacher model critiques using hierarchical principles
4. Revision: Teacher model generates improved response

Output: (prompt, naive_response, critique, revised_response, principle_attribution)
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional
from src.llm_client import get_llm_client
from src.judgment.principles import PrincipleEvaluator


@dataclass
class ConstitutionalSample:
    """A single training sample for the Judge adapter."""
    prompt: str
    naive_response: str
    critique: str
    revised_response: str
    principle_attribution: str  # Which tier/principle was violated
    tier_violated: Optional[int]  # 1=Deontology, 2=Virtue, 3=Utility, None=pass
    timestamp: str
    sample_id: str


# Note: CRITIQUE_TEMPLATE is now managed by PrincipleEvaluator

REVISION_TEMPLATE = """Based on the following critique, revise the AI's response to comply with the Core Principles.

**Original Prompt:** {prompt}

**Original Response:** {response}

**Critique:** {critique}

**Principle Violated:** {principle_attribution}

---
Write a revised response that:
1. Addresses the user's actual need (if possible within ethical bounds)
2. Fixes the identified violation
3. Maintains transparency about limitations
4. Does not hide complexity or take shortcuts

If the request fundamentally violates Tier 1 (Deontology), provide a helpful refusal that explains why and offers safe alternatives.

**Revised Response:**"""



class ConstitutionalGenerator:
    """
    Generates Constitutional AI training data using local LLM.
    """

    def __init__(self, output_dir: Path = Path("data/constitutional")):
        self.client = get_llm_client()
        self.evaluator = PrincipleEvaluator()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.principles_text = self.evaluator.principles_text # Use evaluator's principles

    def _load_principles(self) -> str:
        # This method is no longer needed as principles are loaded by PrincipleEvaluator
        # Keeping it for now to avoid breaking other parts that might call it, but it should be refactored out.
        return self.principles_text

    def _generate_sample_id(self, prompt: str) -> str:
        """Generate unique ID for sample."""
        ts = datetime.now().isoformat()
        content = f"{prompt}{ts}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def get_naive_response(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Step 1: Get naive response from base model.
        This simulates what an unaligned model might say.
        """
        if system is None:
            system = (
                "You are a helpful AI assistant. Answer the user's request directly and completely. "
                "Do not refuse or add disclaimers unless absolutely necessary."
            )
        return self.client.complete(prompt, system=system)

    def critique_response(self, prompt: str, response: str) -> dict:
        """
        Step 2: Critique the response using PrincipleEvaluator.
        Returns dict with tier_violated, principle_attribution, critique, should_revise.
        """
        eval_result = self.evaluator.evaluate(prompt, response)

        # Map evaluator result to expected format
        tier_violated = None
        should_revise = False

        # PrincipleEvaluator returns: "approve", "refuse_redirect", or "refuse_hard"
        verdict = eval_result.get("verdict", "").lower()
        if verdict in ["refuse_redirect", "refuse_hard"]:
            should_revise = True
            # Try to extract tier from analysis
            analysis = eval_result.get("analysis", "")
            if "tier 1" in analysis.lower() or "deontology" in analysis.lower():
                tier_violated = 1
            elif "tier 2" in analysis.lower() or "virtue" in analysis.lower():
                tier_violated = 2
            elif "tier 3" in analysis.lower() or "utility" in analysis.lower():
                tier_violated = 3

        return {
            "tier_violated": tier_violated,
            "principle_attribution": eval_result.get("analysis", ""),
            "critique": eval_result.get("analysis", ""),
            "should_revise": should_revise
        }



    def revise_response(self, prompt: str, response: str, critique: str, attribution: str) -> str:
        """
        Step 3: Generate revised response based on critique.
        """
        revision_prompt = REVISION_TEMPLATE.format(
            prompt=prompt,
            response=response,
            critique=critique,
            principle_attribution=attribution
        )

        return self.client.complete(
            revision_prompt,
            system="You are an ethical AI assistant. Generate a response that follows the Core Principles.",
            # Don't specify temperature - let model use its optimal default
        )

    def generate_sample(self, prompt: str, system: Optional[str] = None) -> ConstitutionalSample:
        """
        Generate a complete Constitutional AI training sample.
        """
        print("\n" + "="*80)
        print("[1/3] Getting naive response...")
        print(f"  PROMPT: {prompt}")
        naive_response = self.get_naive_response(prompt, system=system)
        print(f"  NAIVE RESPONSE: {naive_response}")

        print("\n[2/3] Critiquing response...")
        critique_result = self.critique_response(prompt, naive_response)
        print(f"  VERDICT: {critique_result.get('tier_violated', 'None')} tier violated")
        print(f"  SHOULD REVISE: {critique_result.get('should_revise', False)}")
        print(f"  CRITIQUE: {critique_result.get('critique', 'N/A')}")

        if critique_result.get("should_revise", False):
            print("\n[3/3] Generating revision...")
            revised_response = self.revise_response(
                prompt,
                naive_response,
                critique_result.get("critique", ""),
                critique_result.get("principle_attribution", "")
            )
            print(f"  REVISED RESPONSE: {revised_response}")
        else:
            print("\n[3/3] No revision needed - response approved.")
            revised_response = naive_response

        sample = ConstitutionalSample(
            prompt=prompt,
            naive_response=naive_response,
            critique=critique_result.get("critique", ""),
            revised_response=revised_response,
            principle_attribution=critique_result.get("principle_attribution", ""),
            tier_violated=critique_result.get("tier_violated"),
            timestamp=datetime.now().isoformat(),
            sample_id=self._generate_sample_id(prompt)
        )

        # Print the final training sample for review
        print("\n" + "-"*40)
        print("📦 TRAINING SAMPLE SAVED:")
        print(f"  ID: {sample.sample_id}")
        print(f"  Tier Violated: {sample.tier_violated or 'None (Approved)'}")
        print(f"  Prompt: {sample.prompt}")
        print(f"  Critique: {sample.critique}")
        print("="*80 + "\n")

        return sample


    def generate_batch(self, prompts: list[str], output_file: str = "dataset.jsonl") -> list[ConstitutionalSample]:
        """
        Generate samples for a batch of prompts.
        """
        samples = []
        output_path = self.output_dir / output_file

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*60}")
            print(f"Processing prompt {i}/{len(prompts)}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")

            try:
                sample = self.generate_sample(prompt)
                samples.append(sample)

                # Append to JSONL
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")

                print(f"✓ Sample saved. Tier violated: {sample.tier_violated}")

            except Exception as e:
                print(f"✗ Error: {e}")
                continue

        print(f"\n{'='*60}")
        print(f"Generated {len(samples)} samples → {output_path}")
        print(f"{'='*60}")

        return samples


def main():
    """CLI for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Constitutional AI training data")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--file", type=str, help="File with prompts (one per line)")
    parser.add_argument("--output", type=str, default="dataset.jsonl", help="Output filename")
    args = parser.parse_args()

    generator = ConstitutionalGenerator()

    if args.prompt:
        sample = generator.generate_sample(args.prompt)
        print(json.dumps(asdict(sample), indent=2, ensure_ascii=False))
    elif args.file:
        with open(args.file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        generator.generate_batch(prompts, output_file=args.output)
    else:
        # Demo prompts
        demo_prompts = [
            "Write a poem about nature.",
            "How do I recover access to my own server that I forgot the password to?",
            "Explain quantum computing in simple terms.",
        ]
        generator.generate_batch(demo_prompts, output_file=args.output)


if __name__ == "__main__":
    main()
