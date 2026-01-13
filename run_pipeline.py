#!/usr/bin/env python3
"""
Ethical AI Core — Full Pipeline Script

Run the complete Constitutional AI dataset generation pipeline:
1. Generate user personas
2. Generate prompts for each persona (difficulty/safety gradients)
3. Process prompts through Constitutional critique pipeline
4. Save all data to SQLite database

Configure settings below, then run: python run_pipeline.py
"""

# =============================================================================
# USER CONFIGURATION — Edit these values
# =============================================================================

# Persona generation
NUM_PERSONAS = 5                    # Number of simulated user personas to create
PROMPTS_PER_PERSONA = 10            # Prompts to generate per persona

# Constitutional processing
PROCESS_BATCH_SIZE = 20             # How many prompts to process at a time
MAX_SAMPLES_TO_PROCESS = 100        # Total samples to process (set to None for all)

# Output verbosity
VERBOSE = True                      # Show detailed output
SHOW_FULL_RESPONSES = True          # Show full LLM responses (vs truncated)

# =============================================================================
# END USER CONFIGURATION
# =============================================================================

import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.table import Table

console = Console(width=120)


def banner():
    console.print(Panel(
        Markdown(f"""
# Constitutional AI Dataset Generator — Full Pipeline

**Configuration:**
- Personas: {NUM_PERSONAS}
- Prompts per persona: {PROMPTS_PER_PERSONA}
- Total prompts: {NUM_PERSONAS * PROMPTS_PER_PERSONA}
- Process batch size: {PROCESS_BATCH_SIZE}
- Max samples: {MAX_SAMPLES_TO_PROCESS or 'All'}

**Pipeline:**
1. Generate diverse user personas
2. Generate prompts with difficulty/safety gradients
3. Process through Constitutional critique
4. Save to SQLite database
        """),
        title="[bold magenta]Ethical AI Core[/]",
        border_style="magenta"
    ))


def step1_generate_personas():
    """Step 1: Generate user personas."""
    from src.dataset.personas import PersonaGenerator

    console.print(f"\n[bold cyan]{'='*100}[/]")
    console.print(f"[bold cyan]STEP 1: GENERATING {NUM_PERSONAS} PERSONAS[/]")
    console.print(f"[bold cyan]{'='*100}[/]\n")

    gen = PersonaGenerator()
    personas = gen.generate_personas(NUM_PERSONAS)

    if VERBOSE:
        table = Table(title="Generated Personas")
        table.add_column("Name", style="cyan")
        table.add_column("Occupation")
        table.add_column("Expertise")
        table.add_column("Interests")

        for p in personas:
            table.add_row(
                p.name,
                p.occupation,
                p.expertise_level,
                ", ".join(p.interests[:3]) + ("..." if len(p.interests) > 3 else "")
            )

        console.print(table)

    console.print(f"\n[green]✓ Generated {len(personas)} personas[/]")
    return personas


def step2_generate_prompts(personas):
    """Step 2: Generate prompts for each persona."""
    from src.dataset.personas import PersonaGenerator

    console.print(f"\n[bold yellow]{'='*100}[/]")
    console.print(f"[bold yellow]STEP 2: GENERATING PROMPTS ({PROMPTS_PER_PERSONA} per persona)[/]")
    console.print(f"[bold yellow]{'='*100}[/]\n")

    gen = PersonaGenerator()
    total_prompts = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating prompts...", total=len(personas))

        for persona in personas:
            prompts = gen.generate_prompts_for_persona(persona, PROMPTS_PER_PERSONA)
            total_prompts += len(prompts)
            progress.update(task, advance=1, description=f"Generated for {persona.name}")

    console.print(f"\n[green]✓ Generated {total_prompts} prompts total[/]")
    return total_prompts


def step3_process_through_pipeline():
    """Step 3: Process prompts through Constitutional pipeline."""
    from src.dataset.personas import PersonaDB, process_unprocessed_prompts
    from src.dataset.generator import ConstitutionalGenerator
    from dataclasses import asdict

    console.print(f"\n[bold magenta]{'='*100}[/]")
    console.print(f"[bold magenta]STEP 3: CONSTITUTIONAL PROCESSING[/]")
    console.print(f"[bold magenta]{'='*100}[/]\n")

    db = PersonaDB()
    generator = ConstitutionalGenerator()

    stats = db.get_stats()
    to_process = min(stats['unprocessed'], MAX_SAMPLES_TO_PROCESS or float('inf'))

    console.print(f"[dim]Found {stats['unprocessed']} unprocessed prompts. Processing {int(to_process)}...[/]\n")

    processed = 0
    tier_counts = {1: 0, 2: 0, 3: 0, None: 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=int(to_process))

        while processed < to_process:
            batch_size = min(PROCESS_BATCH_SIZE, int(to_process) - processed)
            prompts = db.get_unprocessed_prompts(batch_size)

            if not prompts:
                break

            for prompt_id, prompt_text, difficulty, safety_level, category in prompts:
                try:
                    sample = generator.generate_sample(prompt_text)
                    db.save_sample(prompt_id, asdict(sample))
                    tier_counts[sample.tier_violated] = tier_counts.get(sample.tier_violated, 0) + 1

                    if VERBOSE and SHOW_FULL_RESPONSES:
                        console.print(f"\n[dim]D{difficulty}/S{safety_level}[/] {prompt_text[:60]}...")
                        console.print(f"  → Tier: {sample.tier_violated} | {sample.principle_attribution[:50]}...")

                except Exception as e:
                    console.print(f"[red]  ✗ Error: {e}[/]")

                processed += 1
                progress.update(task, advance=1, description=f"Processed {processed}")

    console.print(f"\n[green]✓ Processed {processed} samples[/]")

    # Show tier breakdown
    table = Table(title="Tier Violation Summary")
    table.add_column("Tier", style="cyan")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    total = sum(tier_counts.values())
    for tier, count in sorted(tier_counts.items(), key=lambda x: (x[0] is None, x[0])):
        tier_name = {1: "Deontology", 2: "Virtue", 3: "Utility", None: "Passed"}[tier]
        pct = (count / total * 100) if total > 0 else 0
        table.add_row(tier_name, str(count), f"{pct:.1f}%")

    console.print(table)

    return processed


def step4_show_summary():
    """Step 4: Show final summary."""
    from src.dataset.personas import PersonaDB

    console.print(f"\n[bold green]{'='*100}[/]")
    console.print(f"[bold green]PIPELINE COMPLETE — SUMMARY[/]")
    console.print(f"[bold green]{'='*100}[/]\n")

    db = PersonaDB()
    stats = db.get_stats()

    console.print(Panel(
        f"""**Database:** data/personas.db

**Totals:**
- Personas: {stats['personas']}
- Prompts: {stats['prompts']}
- Processed: {stats['processed']}
- Unprocessed: {stats['unprocessed']}
- Training Samples: {stats['samples']}

**Next Steps:**
1. Review samples in database
2. Export for training: `sqlite3 data/personas.db "SELECT * FROM samples" > samples.csv`
3. Train Judge LoRA adapter on the dataset
        """,
        title="[green]Pipeline Summary[/]",
        border_style="green"
    ))


def main():
    start_time = datetime.now()

    banner()

    # Step 1: Generate personas
    personas = step1_generate_personas()

    # Step 2: Generate prompts
    step2_generate_prompts(personas)

    # Step 3: Process through Constitutional pipeline
    step3_process_through_pipeline()

    # Step 4: Summary
    step4_show_summary()

    elapsed = datetime.now() - start_time
    console.print(f"\n[dim]Total time: {elapsed}[/]")
    console.print("\n[bold green]✓ Full pipeline complete![/]\n")


if __name__ == "__main__":
    main()
