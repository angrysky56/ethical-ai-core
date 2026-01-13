#!/usr/bin/env python3
"""
Ethical AI Core — Constitutional AI Demo

Demonstrates the actual dataset generation pipeline:
1. Prompt → Base Model (naive response)
2. Response → Critique (hierarchical evaluation)
3. Critique → Revision (principled correction)

This generates training data for the Judge adapter.
"""
import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

console = Console(width=120)  # Wider output


def run_constitutional_pipeline(prompt: str):
    """Run the full Constitutional AI pipeline on a single prompt."""
    from src.dataset.generator import ConstitutionalGenerator
    from dataclasses import asdict

    console.print("\n" + "═" * 100)
    console.print(Panel(
        prompt,
        title="[bold cyan]INPUT PROMPT[/]",
        border_style="cyan",
        expand=True
    ))

    generator = ConstitutionalGenerator()

    # Step 1: Naive response
    console.print("\n" + "─" * 100)
    console.print("[bold yellow]STEP 1: BASE MODEL (NAIVE RESPONSE)[/]")
    console.print("[dim]Getting unfiltered response from local LLM...[/]\n")
    naive = generator.get_naive_response(prompt)
    console.print(Panel(
        Markdown(naive),
        title="[yellow]Naive Response (Unfiltered)[/]",
        border_style="yellow",
        expand=True
    ))

    # Step 2: Critique
    console.print("\n" + "─" * 100)
    console.print("[bold magenta]STEP 2: CONSTITUTIONAL CRITIQUE[/]")
    console.print("[dim]Evaluating against hierarchical principles...[/]\n")
    critique_result = generator.critique_response(prompt, naive)

    tier = critique_result.get("tier_violated")
    tier_names = {1: "Deontology", 2: "Virtue Ethics", 3: "Servant Utility", None: "None (Passed)"}
    tier_str = f"Tier {tier}: {tier_names.get(tier, 'Unknown')}" if tier else "None (Passed)"
    tier_color = {1: "red", 2: "yellow", 3: "blue", None: "green"}[tier]

    console.print(f"[bold {tier_color}]Tier Violated: {tier_str}[/]\n")

    console.print("[bold]Principle Attribution:[/]")
    console.print(critique_result.get("principle_attribution", "N/A"))
    console.print()

    console.print("[bold]Full Critique:[/]")
    console.print(Panel(
        critique_result.get("critique", "No critique generated"),
        border_style=tier_color,
        expand=True
    ))

    # Step 3: Revision (if needed)
    console.print("\n" + "─" * 100)
    if critique_result.get("should_revise", False):
        console.print("[bold green]STEP 3: PRINCIPLED REVISION[/]")
        console.print("[dim]Generating corrected response based on critique...[/]\n")
        revised = generator.revise_response(
            prompt, naive,
            critique_result.get("critique", ""),
            critique_result.get("principle_attribution", "")
        )
        console.print(Panel(
            Markdown(revised),
            title="[green]Revised Response (Principled)[/]",
            border_style="green",
            expand=True
        ))
    else:
        console.print("[bold green]STEP 3: NO REVISION NEEDED[/]")
        console.print("[dim]Response passed constitutional check.[/]")

    console.print("\n" + "═" * 100)
    console.print("[bold]✓ Sample ready for Judge adapter training.[/]\n")

    # Show the full sample that would be saved
    full_sample = generator.generate_sample(prompt)

    # SAVE TO DB
    from src.dataset.personas import SampleDB
    sample_db = SampleDB()
    # We don't have a prompt ID here, so we generate a random one or hash
    import hashlib
    p_id = hashlib.md5(prompt.encode()).hexdigest()[:8]
    # We pass the prompt text twice (once as ID for now/dummy, once as text)
    # Actually wait, save_sample(prompt_id, prompt_text, sample_dict)
    sample_db.save_sample(p_id, prompt, asdict(full_sample))

    console.print("[bold cyan]SAVED SAMPLE TO DB (JSON):[/]")
    console.print(Panel(
        Syntax(json.dumps(asdict(full_sample), indent=2, ensure_ascii=False), "json", theme="monokai", word_wrap=True),
        title="[cyan]Saved to SampleDB[/]",
        border_style="cyan",
        expand=True
    ))


def generate_dataset(count: int = 5):
    """Generate a batch of Constitutional training samples."""
    from src.dataset.generator import ConstitutionalGenerator
    from src.dataset.prompts import get_all_prompts
    import random
    import hashlib
    from dataclasses import asdict
    from src.dataset.personas import SampleDB, RunManager

    sample_db = SampleDB()

    console.print(Panel(
        f"Generating {count} Constitutional AI training samples...",
        title="[cyan]Dataset Generation[/]",
        border_style="cyan"
    ))

    generator = ConstitutionalGenerator()
    prompts = random.sample(get_all_prompts(), min(count, len(get_all_prompts())))

    for i, prompt in enumerate(prompts, 1):
        console.print(f"\n[bold cyan]{'='*100}[/]")
        console.print(f"[bold cyan]SAMPLE {i}/{count}[/]")
        console.print(f"[bold cyan]{'='*100}[/]")

        try:
            sample = generator.generate_sample(prompt)

            # Show full sample
            console.print(f"\n[bold]Prompt:[/] {sample.prompt}")

            console.print("\n[bold yellow]Naive Response:[/]")
            console.print(Panel(Markdown(sample.naive_response), border_style="yellow", expand=True))

            tier_color = {1: "red", 2: "yellow", 3: "blue", None: "green"}[sample.tier_violated]
            console.print(f"\n[bold {tier_color}]Tier Violated: {sample.tier_violated}[/]")
            console.print(f"[bold]Attribution: {sample.principle_attribution}[/]")

            console.print("\n[bold magenta]Critique:[/]")
            console.print(Panel(sample.critique, border_style="magenta", expand=True))

            if sample.revised_response != sample.naive_response:
                console.print("\n[bold green]Revised Response:[/]")
                console.print(Panel(Markdown(sample.revised_response), border_style="green", expand=True))

            console.print(f"\n[dim]Sample ID: {sample.sample_id} | Timestamp: {sample.timestamp}[/]")

            # Save to DB
            sample_db.save_sample(
                hashlib.md5(prompt.encode()).hexdigest()[:8],
                prompt,
                asdict(sample)
            )

        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/]")
            continue

    console.print(f"\n[bold green]✓ Generated {count} samples → {RunManager.get_run_dir()}[/]")


def demo_personas(num_personas: int = 2, prompts_per_persona: int = 5):
    """Demo the persona-based generation pipeline."""
    from src.dataset.personas import PersonaGenerator

    console.print(Panel(
        Markdown("""
# Persona-Based Prompt Generation

**Pipeline:**
1. Generate diverse user personas (age, occupation, interests, expertise)
2. For each persona, generate prompts across difficulty/safety gradients
3. Store in SQLite database for scale
4. Process through Constitutional pipeline
        """),
        title="[bold magenta]Persona Generation Demo[/]",
        border_style="magenta"
    ))

    gen = PersonaGenerator()

    # Step 1: Generate personas
    console.print(f"\n[bold cyan]{'='*100}[/]")
    console.print(f"[bold cyan]STEP 1: GENERATING {num_personas} PERSONAS[/]")
    console.print(f"[bold cyan]{'='*100}[/]\n")

    personas = gen.generate_personas(num_personas)

    for persona in personas:
        console.print(Panel(
            f"""**Name:** {persona.name}
**Age Range:** {persona.age_range}
**Occupation:** {persona.occupation}
**Interests:** {', '.join(persona.interests)}
**Expertise:** {persona.expertise_level}
**Style:** {persona.communication_style}
**Background:** {persona.background}""",
            title=f"[cyan]Persona: {persona.name}[/]",
            border_style="cyan"
        ))

    # Step 2: Generate prompts for each persona
    console.print(f"\n[bold yellow]{'='*100}[/]")
    console.print("[bold yellow]STEP 2: GENERATING PROMPTS FOR EACH PERSONA[/]")
    console.print(f"[bold yellow]{'='*100}[/]\n")

    for persona in personas:
        console.print(f"\n[bold]Generating prompts for {persona.name}...[/]")
        prompts = gen.generate_prompts_for_persona(persona, prompts_per_persona)

        for p in prompts:
            safety_color = {1: "green", 2: "green", 3: "yellow", 4: "red", 5: "red"}.get(p.safety_level, "white")
            console.print(f"  [dim]D{p.difficulty}/S{p.safety_level}[/] [{safety_color}]{p.category}[/]: {p.prompt[:80]}...")

    # Show stats
    console.print(f"\n[bold green]{'='*100}[/]")
    console.print("[bold green]DATABASE STATS[/]")
    console.print(f"[bold green]{'='*100}[/]\n")

    from src.dataset.personas import get_run_stats
    stats = get_run_stats()
    console.print(Panel(
        Syntax(json.dumps(stats, indent=2), "json", theme="monokai"),
        title="[green]data/runs/demo_run[/]",
        border_style="green"
    ))

    console.print("\n[dim]Run 'python -m src.dataset.personas --process N' to process through Constitutional pipeline[/]")


def main():
    parser = argparse.ArgumentParser(description="Constitutional AI Demo")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--generate", type=int, metavar="N", help="Generate N training samples")
    parser.add_argument("--personas", action="store_true", help="Demo persona-based generation")
    args = parser.parse_args()

    # Set up Demo Run
    from src.dataset.personas import RunManager
    RunManager.set_run("demo_run")
    console.print(f"[dim]Using Run Directory: {RunManager.get_run_dir()}[/]")

    console.print(Panel(
        Markdown("""
# Constitutional AI Dataset Generator

**Pipeline:**
1. **Base Model** → Generates naive (unfiltered) response
2. **Critique** → Evaluates against hierarchical principles
3. **Revision** → Generates corrected response

**Output:** `(prompt, naive, critique, revised)` tuples for Judge training.

**All responses shown in full — no truncation.**
        """),
        title="[bold magenta]Ethical AI Core[/]",
        border_style="magenta"
    ))

    if args.prompt:
        run_constitutional_pipeline(args.prompt)
    elif args.generate:
        generate_dataset(args.generate)
    elif args.personas:
        demo_personas(num_personas=2, prompts_per_persona=5)
    else:
        # Demo with example prompt
        run_constitutional_pipeline(
            "Is this code correct? def add(a, b): return a - b"
        )

    console.print("\n[bold green]✓ Demo complete![/]\n")


if __name__ == "__main__":
    main()

