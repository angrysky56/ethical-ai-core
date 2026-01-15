#!/usr/bin/env python3
"""
Ethical AI Core — Web UI

A Gradio-based interface for:
1. Dataset Generation (personas, prompts, Constitutional processing)
2. Adapter Training (DDL)
3. Chat Interface (with Judge system)
4. Settings (seed, model, provider)

Run: python ui.py
"""
import gradio as gr
import json
import os
from typing import Optional
from src.dataset.personas import RunManager

from dotenv import load_dotenv

# Load environment config
load_dotenv()

# =============================================================================
# Dynamic Settings State
# =============================================================================
# Provider-specific defaults
PROVIDER_DEFAULTS = {
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    },
    "openrouter": {
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
    },
    "lmstudio": {
        "base_url": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "model": os.getenv("LMSTUDIO_MODEL", "local-model"),
    },
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    },
}

def get_provider_config(provider: str) -> dict:
    """Get config for a specific provider."""
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["ollama"])

class AppState:
    """Mutable app state for settings."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    seed = int(os.getenv("GENERATION_SEED", "0")) or None
    timeout = int(os.getenv("LLM_TIMEOUT", "1200"))
    batch_size = int(os.getenv("BATCH_SIZE", "5"))

    # Load model/base_url based on current provider
    _config = get_provider_config(provider)
    model = _config["model"]
    base_url = _config["base_url"]

state = AppState()

def update_settings(provider: str, model: str, base_url: str, seed: str, batch_size: int):
    """Update app settings dynamically."""
    state.provider = provider
    state.model = model
    state.base_url = base_url
    state.seed = int(seed) if seed.strip() else None
    state.batch_size = int(batch_size)

    # Update environment so LLMClient picks it up
    os.environ["LLM_PROVIDER"] = provider

    if provider == "ollama":
        os.environ["OLLAMA_MODEL"] = model
        os.environ["OLLAMA_BASE_URL"] = base_url
    elif provider == "openrouter":
        os.environ["OPENROUTER_MODEL"] = model
        os.environ["OPENROUTER_BASE_URL"] = base_url
    elif provider == "lmstudio":
        os.environ["LMSTUDIO_MODEL"] = model
        os.environ["LMSTUDIO_BASE_URL"] = base_url
    elif provider == "openai":
        os.environ["OPENAI_MODEL"] = model
        os.environ["OPENAI_BASE_URL"] = base_url

    if state.seed:
        os.environ["GENERATION_SEED"] = str(state.seed)
    elif "GENERATION_SEED" in os.environ:
        del os.environ["GENERATION_SEED"]

    # Reload the config module
    import importlib
    import src.config
    importlib.reload(src.config)

    # Reset the LLM client singleton
    import src.llm_client
    src.llm_client._client = None

    return f"✓ Settings updated: {provider}/{model} (seed={'random' if state.seed is None else state.seed})"

def get_current_settings():
    """Get current settings for display."""
    return state.provider, state.model, state.base_url, str(state.seed or ""), state.timeout


def get_available_runs(include_new=True):
    """List all available dataset runs."""
    try:
        from src.dataset.personas import RunManager
        runs = RunManager.list_runs()

        # Add creation option at top
        if include_new:
            return ["[Create New Run]"] + runs
        return runs
    except ImportError:
        return ["[Create New Run]"] if include_new else []

def set_active_run(run_id):
    """Switch active database run."""
    from src.dataset.personas import RunManager

    if run_id == "[Create New Run]":
        new_id = RunManager.new_run()
        # Return updated stats AND updated dropdown
        choices = get_available_runs()
        return format_stats(), gr.update(choices=choices, value=new_id)

    RunManager.set_run(run_id)
    # Return stats and keep dropdown as is
    return format_stats(), gr.update()

def get_run_choices_with_stats(include_new=False):
    """Get run choices with P/Q/S counts."""
    from src.dataset.personas import PersonaDB, PromptDB, SampleDB
    try:
        raw_runs = get_available_runs(include_new=include_new)
        choices = []
        for run_id in raw_runs:
            if run_id == "[Create New Run]":
                choices.append(run_id)
                continue

            try:
                p_c = PersonaDB(run_id=run_id).count()
                q_c = PromptDB(run_id=run_id).count()
                s_c = SampleDB(run_id=run_id).count()
                label = f"{run_id} ({p_c}P, {q_c}Q, {s_c}S)"
                choices.append((label, run_id))
            except Exception:
                choices.append((run_id, run_id))
        return choices
    except Exception as e:
        print(f"Error getting run stats: {e}")
        return get_available_runs(include_new=include_new)


def delete_run_action(run_id):
    """Delete the specified run."""
    from src.dataset.personas import RunManager
    if not run_id or run_id == "[Create New Run]":
        return "Invalid run selected", gr.update(), gr.update()

    success = RunManager.delete_run(run_id)
    if success:
        # Update all selectors
        choices_w_stats = get_run_choices_with_stats()
        def get_val(ch):
            if not ch: return None
            if isinstance(ch[0], tuple): return ch[0][1]
            return ch[0]

        new_val = get_val(choices_w_stats)

        return (
            f"Deleted run {run_id}",
            gr.update(choices=choices_w_stats, value=new_val), # Training tab
            gr.update(choices=choices_w_stats, value=None) # Settings tab
        )
    return f"Failed to delete run {run_id}", gr.update(), gr.update()


def rename_run_action(run_id, new_name):
    """Rename the specified run."""
    from src.dataset.personas import RunManager
    if not run_id or not new_name or run_id == "[Create New Run]":
        return "Invalid input", gr.update()

    success = RunManager.rename_run(run_id, new_name)
    if success:
        clean_name = new_name.replace(" ", "_").strip() # Rough estimate of what RunManager did
        # Actually RunManager cleans it. We should probably list runs to be sure.
        return f"Renamed to {clean_name}", gr.update(choices=get_available_runs(), value=None)
    return "Failed to rename run", gr.update()

def get_personas_list(run_id, db_filename="personas.db"):
    """Get list of persona names from specified DB."""
    from src.dataset.personas import PersonaDB
    try:
        if not run_id or run_id == "[Create New Run]":
            return []

        # Determine actual Run ID if passed as tuple string or similar?
        # Gradio dropdown value is just the ID string usually if we set it right.

        db = PersonaDB(run_id=str(run_id), db_filename=db_filename)
        personas = db.get_all_personas()
        return sorted([p['name'] for p in personas])
    except Exception as e:
        print(f"Error listing personas: {e}")
        return []

def get_persona_system_prompt(run_id, db_filename, persona_name):
    """Get system prompt for a specific persona."""
    from src.dataset.personas import PersonaDB
    try:
         db = PersonaDB(run_id, db_filename=db_filename)
         all_p = db.get_all_personas()
         target = next((p for p in all_p if p['name'] == persona_name), None)

         if not target:
             return "You are a helpful AI assistant."
         return f"You are {target['name']}, a {target['age_range']} year old {target['occupation']}.\\nInterests: {', '.join(target['interests'])}\\nBackground: {target['background']}\\nStyle: {target['communication_style']}"
    except Exception:
         return "You are a helpful AI assistant."



def get_db_stats():
    """Get current database statistics."""
    try:
        from src.dataset.personas import get_run_stats
        return get_run_stats()
    except Exception as e:
        return {"error": str(e)}

def format_stats():
    """Format stats for display."""
    s = get_db_stats()
    return f"📊 **{s.get('personas', 0)}** Personas | **{s.get('prompts', 0)}** Prompts | **{s.get('samples', 0)}** Samples"


def refresh_models_list():
    """List available models from current provider."""
    from src.llm_client import get_llm_client
    client = get_llm_client()
    try:
        models = client.list_models()
        return gr.update(choices=models, value=models[0] if models else None)
    except Exception:
        return gr.update(choices=[])

def get_initial_models():
    """Get models list for initial dropdown population."""
    from src.llm_client import get_llm_client
    try:
        client = get_llm_client()
        return client.list_models()
    except Exception:
        return []


def fetch_openrouter_models():
    """Fetch OpenRouter models with pricing, return as formatted HTML."""
    from src.llm_client import LLMClient

    client = LLMClient(provider="openrouter")
    providers = client.list_openrouter_models_with_pricing()

    if not providers:
        return "❌ Failed to fetch models. Check your OpenRouter API key.", gr.update(choices=[])

    # Build HTML with collapsible sections
    html_parts = [
        "<style>",
        ".or-provider { margin: 8px 0; }",
        ".or-provider details { background: #1a1a2e; border-radius: 8px; padding: 8px 12px; }",
        ".or-provider summary { cursor: pointer; font-weight: bold; color: #fff; }",
        ".or-model { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #333; font-size: 13px; }",
        ".or-model:hover { background: #252540; }",
        ".or-name { flex: 2; color: #88c0d0; cursor: pointer; }",
        ".or-name:hover { text-decoration: underline; }",
        ".or-price { flex: 1; text-align: right; color: #a3be8c; }",
        ".or-free { color: #88ff88; font-weight: bold; }",
        ".or-ctx { flex: 0.5; text-align: right; color: #888; font-size: 11px; }",
        "</style>"
    ]

    all_models = []  # For dropdown
    free_count = 0

    for provider_name, models in providers.items():
        provider_free = sum(1 for m in models if m["is_free"])
        free_count += provider_free

        html_parts.append('<div class="or-provider">')
        html_parts.append(f'<details><summary>📦 {provider_name} ({len(models)} models, {provider_free} free)</summary>')

        for m in models:
            all_models.append((f"{m['name']} | ${m['input_price']:.4f}/${m['output_price']:.4f}", m['id']))

            if m["is_free"]:
                price_str = '<span class="or-free">FREE</span>'
            else:
                price_str = f"${m['input_price']:.4f} / ${m['output_price']:.4f}"

            ctx = f"{m['context_length']//1000}K" if m['context_length'] else "?"

            html_parts.append(
                f'<div class="or-model" onclick="navigator.clipboard.writeText(\'{m["id"]}\'); this.style.background=\'#4a4; setTimeout(() => this.style.background=\'\', 200);">'
                f'<span class="or-name">{m["name"]}</span>'
                f'<span class="or-ctx">{ctx}</span>'
                f'<span class="or-price">{price_str}</span>'
                f'</div>'
            )

        html_parts.append('</details></div>')

    total_models = sum(len(m) for m in providers.values())
    header = f"**{len(providers)} providers** | **{total_models} models** | **{free_count} free** | *Click model to copy ID*"

    return header + "\n\n" + "".join(html_parts), gr.update(choices=all_models)


# =============================================================================
# TAB 1: Dataset Generation
# =============================================================================

def generate_personas(num_personas: int, mode: str = "simple", db_filename: str = "personas.db", progress=gr.Progress()):
    """Generate user personas."""
    from src.dataset.personas import PersonaGenerator, PersonaDB
    from src.config import GLOBAL_SEED

    progress(0, desc="Initializing...")
    gen = PersonaGenerator(db_filename=db_filename)

    progress(0.2, desc=f"Generating {num_personas} personas (seed={GLOBAL_SEED}, mode={mode})...")
    personas = gen.generate_personas(num_personas, mode=mode)  # Uses GLOBAL_SEED by default

    output = [f"**Generated {len(personas)} new personas (seed={GLOBAL_SEED}):**\n"]
    for p in personas:
        output.append(f"### {p.name}\n- **Occupation:** {p.occupation}\n- **Age:** {p.age_range}\n- **Interests:** {', '.join(p.interests)}\n- **Expertise:** {p.expertise_level}\n- **Style:** {p.communication_style}\n- **Background:** {p.background}\n")

    # Also show total in DB
    db = PersonaDB(db_filename=db_filename)
    all_personas = db.get_all_personas()
    output.append(f"\n---\n**Total personas in database:** {len(all_personas)}")

    # Get current run for auto-cascade
    from src.dataset.personas import RunManager
    current_run = RunManager.get_current_run()
    choices = get_run_choices_with_stats()

    return "\n".join(output), format_stats(), gr.update(choices=choices, value=current_run)

def generate_simple_personas(num_personas: int, progress=gr.Progress()):
    return generate_personas(num_personas, mode="simple", db_filename="personas.db", progress=progress)

def generate_detailed_personas(num_personas: int, progress=gr.Progress()):
    return generate_personas(num_personas, mode="detailed", db_filename="full_personas.db", progress=progress)


def generate_prompts(prompts_per_persona: int, min_difficulty: int = 1, min_safety: int = 1, personas_source_run: Optional[str] = None, progress=gr.Progress()):
    """Generate prompts using personas from source run."""
    from src.dataset.personas import PersonaGenerator, PersonaDB, Persona

    progress(0, desc="Loading personas...")
    # Use source run for reading personas (always use simple personas.db)
    db = PersonaDB(run_id=personas_source_run, db_filename="personas.db")
    gen = PersonaGenerator()



    persona_dicts = db.get_all_personas()
    personas = [Persona(**p) for p in persona_dicts]

    if not personas:
        return "No personas found. Generate personas first.", format_stats()

    total_prompts = 0
    output = []

    progress(0, desc=f"Generating prompts for {len(personas)} personas...")

    for i, persona in enumerate(personas):
        progress((i) / len(personas), desc=f"Generating for {persona.name}...")

        prompts = gen.generate_prompts_for_persona(
            persona, prompts_per_persona,
            batch_size=state.batch_size,
            min_difficulty=min_difficulty,
            min_safety=min_safety
        )
        total_prompts += len(prompts)

        output.append(f"✓ {persona.name}: {len(prompts)} prompts")

    output.append(f"\\n---\\n**Done! Total new prompts:** {total_prompts}")

    # Get current run for auto-cascade to Step 3
    from src.dataset.personas import RunManager
    current_run = RunManager.get_current_run()
    choices = get_run_choices_with_stats()

    return "\n".join(output), format_stats(), gr.update(choices=choices, value=current_run)


def process_prompts(prompts_source_run: Optional[str] = None, progress=gr.Progress()):
    """Process prompts through Constitutional pipeline."""
    from src.dataset.personas import PromptDB, SampleDB
    from src.dataset.generator import ConstitutionalGenerator
    from dataclasses import asdict

    progress(0, desc="Loading prompts...")

    # Use source run for reading prompts
    prompt_db = PromptDB(run_id=prompts_source_run)
    # Always save to CURRENT run
    sample_db = SampleDB()
    generator = ConstitutionalGenerator()

    prompts = prompt_db.get_unprocessed()  # Process all unprocessed prompts

    if not prompts:
        return "No unprocessed prompts found.", format_stats()

    output = []
    processed = 0
    errors = 0
    for i, (prompt_id, prompt_text, difficulty, safety_level, category, persona_id) in enumerate(prompts):
        progress((i + 1) / len(prompts), desc=f"Processing {i+1}/{len(prompts)}...")

        try:
            sample = generator.generate_sample(prompt_text)
            sample_db.save_sample(prompt_id, prompt_text, asdict(sample))
            prompt_db.mark_processed(prompt_id)
            processed += 1
        except Exception as e:
            errors += 1
            output.append(f"[ERROR] {e}")

    output.insert(0, f"✓ Done! Processed {processed} samples" + (f" ({errors} errors)" if errors else ""))
    return "\\n".join(output), format_stats()


def export_dataset():
    """Export samples to JSONL for training."""
    from src.dataset.personas import SampleDB, RunManager

    db = SampleDB()
    run_dir = RunManager.get_run_dir()
    output_path = run_dir / "training_export.jsonl"

    count = db.export_jsonl(output_path)
    return f"Exported {count} samples to `{output_path}`"


# =============================================================================
# TAB 2: Training (Deep Delta Learning)
# =============================================================================

def start_training(
    method: str,
    epochs: int,
    learning_rate: float,
    num_heads: int,
    rank: int,
    run_id: Optional[str] = None,
    progress=gr.Progress()
):
    """Unified training function for DDL and ReFT methods."""
    from src.training.train_unified import train_ddl, train_reft
    from src.llm_client import get_llm_client
    from src.dataset.personas import SampleDB, RunManager

    progress(0, desc="Loading training data...")

    if run_id:
        RunManager.set_run(run_id)

    # Load samples
    db = SampleDB()
    samples = db.get_all()

    if not samples:
        return "No samples found in dataset. Generate samples first."

    progress(0.02, desc=f"Found {len(samples)} samples. Starting {method.upper()} training...")

    # Get client and create progress callback
    client = get_llm_client()

    def progress_callback(pct, msg):
        progress(pct, desc=msg)

    try:
        if method == "ddl":
            result = train_ddl(
                samples=samples,
                client=client,
                epochs=epochs,
                learning_rate=learning_rate,
                num_heads=num_heads,
                progress_fn=progress_callback
            )
        else:  # reft
            result = train_reft(
                samples=samples,
                client=client,
                epochs=epochs,
                learning_rate=learning_rate,
                rank=rank,
                num_heads=num_heads,
                progress_fn=progress_callback
            )

        return f"""**{result.method.upper()} Training Complete**

**Configuration:**
- Method: {result.method.upper()}
- Epochs: {result.epochs}
- Learning Rate: {learning_rate}
- {'Delta Heads' if method == 'ddl' else 'LoReFT Heads'}: {num_heads}
{f'- Intervention Rank: {rank}' if method == 'reft' else ''}
- Embedding Dimension: {result.embed_dim}
- Training Samples: {result.num_samples}

**Results:**
- Initial Loss: {result.initial_loss:.4f}
- Final Loss: {result.final_loss:.4f}
- Improvement: {result.improvement_pct:.1f}%

**Model saved to:** `{result.model_path}`
"""

    except Exception as e:
        return f"**Training Error:** {e}"


# =============================================================================
# TAB 3: Chat Interface
# =============================================================================

def chat_handler(message: str, history: list, system_prompt: str = "You are a helpful AI assistant."):
    """Chat interface with Constitutional Judge."""
    if not message.strip():
        return history, ""

    from src.llm_client import get_llm_client
    from src.judgment.principles import PrincipleEvaluator

    client = get_llm_client()
    evaluator = PrincipleEvaluator()

    # Step 1: Get base model response
    base_response = client.complete(
        message,
        system=system_prompt,
        temperature=0.7
    )

    # Step 2: Judge evaluates the response
    judgment = evaluator.evaluate(message, base_response)

    # Step 3: Format output based on verdict
    tier = judgment.get('tier_violated')
    verdict = judgment.get('verdict', 'unknown')


    if verdict == 'approve':
        final_response = base_response
        status = "✓ Approved"
    elif verdict == 'refuse_redirect':
        final_response = base_response
        status = f"⚠ Tier {tier} — {judgment.get('principle_attribution', 'Review suggested')}"
    else:
        final_response = "I cannot fulfill this request as it violates core ethical principles."
        status = f"✗ Blocked (Tier {tier})"

    # Format with judge status
    formatted = f"{final_response}\n\n---\n**Judge:** {status}"

    history = history or []
    history.append((message, formatted))
    return history, ""


# =============================================================================
# BUILD THE UI
# =============================================================================

# =============================================================================
# BUILD THE UI
# =============================================================================

# Initialize Active Run (Prevent empty run proliferation)
existing_runs = RunManager.list_runs()
if existing_runs and not RunManager._current_run_id:
    # Resume latest run
    latest_run = existing_runs[0]
    RunManager.set_run(latest_run)
    print(f"[System] Resuming latest active run: {latest_run}")

with gr.Blocks(title="Ethical AI Core") as app:
    gr.Markdown("""
    # 🧠 Ethical AI Core

    **Constitutional AI Dataset Generator & Judge System**
    """)

    with gr.Row():

        gr.Markdown(f"**Provider:** `{state.provider}` | **Model:** `{state.model}` | **Seed:** `{state.seed or 'random'}`")
        stats_text = gr.Markdown(value=format_stats())

    with gr.Tabs():
        # =====================================================================
        # TAB 1: Dataset Generation
        # =====================================================================
        with gr.TabItem("📊 Dataset Generation"):
            gr.Markdown("### Generate Constitutional AI Training Data")

            with gr.Row():
                # COLUMN 1: Step 1 - Generate Personas
                with gr.Column():
                    gr.Markdown("#### Step 1: Generate Personas")
                    personas_run_dd = gr.Dropdown(
                        label="Save to Run",
                        choices=get_run_choices_with_stats(include_new=True),
                        value=lambda: (get_run_choices_with_stats()[0][1] if isinstance(get_run_choices_with_stats()[0], tuple) else get_run_choices_with_stats()[0]) if get_run_choices_with_stats() else None,
                        interactive=True
                    )
                    num_personas = gr.Slider(1, 20, value=5, step=1, label="Number of Personas")
                    gen_personas_btn = gr.Button("Generate Personas", variant="primary")

                # COLUMN 2: Step 2 - Generate Prompts
                with gr.Column():
                    gr.Markdown("#### Step 2: Generate Prompts")
                    personas_source_dd = gr.Dropdown(
                        label="Source Personas Run",
                        choices=get_run_choices_with_stats(),
                        value=lambda: (get_run_choices_with_stats()[0][1] if isinstance(get_run_choices_with_stats()[0], tuple) else get_run_choices_with_stats()[0]) if get_run_choices_with_stats() else None,
                        interactive=True
                    )
                    num_prompts_input = gr.Number(label="Prompts per Persona", value=10, precision=0, minimum=1)
                    with gr.Accordion("⚙️ Prompt Settings", open=False):
                        with gr.Row():
                            min_difficulty_input = gr.Slider(1, 5, value=1, step=1, label="Max Difficulty", scale=1)
                            min_safety_input = gr.Slider(1, 5, value=1, step=1, label="Max Danger", scale=1)
                    generate_prompts_btn = gr.Button("Generate Prompts", variant="primary")


                # COLUMN 3: Step 3 - Generate Samples
                with gr.Column():
                    gr.Markdown("#### Step 3: Generate Samples")
                    prompts_source_dd = gr.Dropdown(
                        label="Source Prompts Run",
                        choices=get_run_choices_with_stats(),
                        value=lambda: (get_run_choices_with_stats()[0][1] if isinstance(get_run_choices_with_stats()[0], tuple) else get_run_choices_with_stats()[0]) if get_run_choices_with_stats() else None,
                        interactive=True
                    )
                    process_btn = gr.Button("Generate Samples", variant="primary")

            output_log = gr.Textbox(label="Output Log", lines=10, interactive=False)

            # Single refresh button for all dropdowns
            def refresh_all_dropdowns():
                choices = get_run_choices_with_stats()
                return gr.update(choices=choices), gr.update(choices=choices), gr.update(choices=choices)

            refresh_all_btn = gr.Button("🔄 Refresh Runs", size="sm")
            refresh_all_btn.click(
                fn=refresh_all_dropdowns,
                outputs=[personas_run_dd, personas_source_dd, prompts_source_dd]
            )

            # Event handlers with auto-cascade
            gen_personas_btn.click(
                fn=generate_simple_personas,
                inputs=[num_personas],
                outputs=[output_log, stats_text, personas_source_dd]  # Cascade to Step 2
            )

            generate_prompts_btn.click(
                fn=generate_prompts,
                inputs=[num_prompts_input, min_difficulty_input, min_safety_input, personas_source_dd],
                outputs=[output_log, stats_text, prompts_source_dd]  # Cascade to Step 3
            )


            process_btn.click(
                fn=process_prompts,
                inputs=[prompts_source_dd],
                outputs=[output_log, stats_text]
            )


        # =====================================================================
        # TAB 2: Detailed Personas
        # =====================================================================
        with gr.TabItem("🎭 Detailed Personas"):
            gr.Markdown("### Detailed Persona Generator")
            gr.Markdown("Generate rich, creative personas with detailed backstories.")

            with gr.Row():
                num_detailed = gr.Slider(1, 10, value=3, step=1, label="Number of Personas")
                gen_detailed_btn = gr.Button("Generate Detailed Personas", variant="primary")

            detailed_output = gr.Markdown()

            gen_detailed_btn.click(
                fn=generate_detailed_personas,
                inputs=[num_detailed],
                outputs=[detailed_output, stats_text]
            )

        # =====================================================================
        # TAB 2: Training (DDL)
        # =====================================================================
        with gr.TabItem("🎓 Training"):
            gr.Markdown("### Train Deep Delta Learning Judge")
            gr.Markdown("""
            **Deep Delta Learning (DDL)** learns surgical corrections:
            - Identifies "bad behavior" directions from flagged samples
            - Trains Delta heads to project bad → good
            - No full fine-tuning required
            """)

            with gr.Row():
                train_run_selector = gr.Dropdown(
                    label="Training Dataset Source",
                    choices=get_run_choices_with_stats(),
                    value=lambda: (get_run_choices_with_stats()[0][1] if isinstance(get_run_choices_with_stats()[0], tuple) else get_run_choices_with_stats()[0]) if get_run_choices_with_stats() else None,
                    interactive=True
                )
                refresh_train_runs_btn = gr.Button("🔄", size="sm", scale=0)

            refresh_train_runs_btn.click(
                fn=lambda: gr.update(choices=get_run_choices_with_stats()),
                outputs=[train_run_selector]
            )

            with gr.Row():
                train_method = gr.Radio(
                    choices=["ddl", "reft"],
                    value="ddl",
                    label="Training Method",
                    info="DDL: rank-1 delta | ReFT: low-rank interventions"
                )

            with gr.Row():
                epochs = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                learning_rate = gr.Number(value=1e-3, label="Learning Rate")

            with gr.Row():
                num_heads = gr.Slider(1, 8, value=4, step=1, label="Heads")
                rank = gr.Slider(1, 16, value=4, step=1, label="Rank (ReFT only)")

            train_btn = gr.Button("🚀 Start Training", variant="primary")
            training_output = gr.Markdown()

            train_btn.click(
                fn=start_training,
                inputs=[train_method, epochs, learning_rate, num_heads, rank, train_run_selector],
                outputs=[training_output]
            )

        # =====================================================================
        # TAB 3: Chat Interface
        # =====================================================================
        with gr.TabItem("💬 Chat"):
            gr.Markdown("### Chat with Constitutional AI")
            gr.Markdown("*Responses are evaluated by the Judge against Core Principles.*")

            with gr.Accordion("🎭 System Persona Configuration", open=False):
                with gr.Row():
                    chat_persona_run_dd = gr.Dropdown(
                        label="Source Run",
                        choices=get_run_choices_with_stats(),
                        value=lambda: (get_run_choices_with_stats()[0][1] if isinstance(get_run_choices_with_stats()[0], tuple) else get_run_choices_with_stats()[0]) if get_run_choices_with_stats() else None,
                        interactive=True
                    )
                    chat_persona_db_dd = gr.Dropdown(label="DB Type", choices=["personas.db", "full_personas.db"], value="full_personas.db", interactive=True)

                with gr.Row():
                    chat_persona_select_dd = gr.Dropdown(label="Select Persona", choices=[], allow_custom_value=False, interactive=True, scale=3)
                    refresh_personas_btn = gr.Button("🔄 Load List", scale=1)

                system_prompt_input = gr.Textbox(label="System Prompt", value="You are a helpful AI assistant.", lines=3)
                load_sys_prompt_btn = gr.Button("⬇ Load Selected Persona into Prompt", size="sm")

                def update_persona_list(run_id, db_filename):
                     choices = get_personas_list(run_id, db_filename)
                     return gr.update(choices=choices, value=choices[0] if choices else None)

                refresh_personas_btn.click(update_persona_list, [chat_persona_run_dd, chat_persona_db_dd], [chat_persona_select_dd])

                def load_prompt_action(run_id, db_filename, persona_name):
                    return get_persona_system_prompt(run_id, db_filename, persona_name)

                load_sys_prompt_btn.click(load_prompt_action, [chat_persona_run_dd, chat_persona_db_dd, chat_persona_select_dd], [system_prompt_input])

            chatbot = gr.Chatbot(label="Constitutional AI", height=450)
            msg = gr.Textbox(label="Your message", placeholder="Type your message here...", lines=2)

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

            msg.submit(chat_handler, [msg, chatbot, system_prompt_input], [chatbot, msg])
            send_btn.click(chat_handler, [msg, chatbot, system_prompt_input], [chatbot, msg])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        # =====================================================================
        # TAB 4: Settings
        # =====================================================================
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### LLM & Generation Settings")
            gr.Markdown("*Changes apply immediately without restart.*")

            with gr.Row():
                provider_input = gr.Dropdown(
                    choices=["ollama", "openrouter", "lmstudio", "openai"],
                    value=state.provider,
                    label="Provider"
                )
                with gr.Row():
                    model_input_settings = gr.Dropdown(
                        choices=get_initial_models(),
                        value=state.model,
                        label="Model Name",
                        allow_custom_value=True,
                        scale=3
                    )
                    refresh_models_settings_btn = gr.Button("🔄", scale=0)

            base_url_input = gr.Dropdown(
                choices=[
                    ("OpenRouter", "https://openrouter.ai/api/v1"),
                    ("Ollama (local)", "http://localhost:11434"),
                    ("LM Studio (local)", "http://localhost:1234/v1"),
                    ("OpenAI", "https://api.openai.com/v1"),
                ],
                value=state.base_url,
                label="Base URL",
                allow_custom_value=True
            )

            # Update models AND base_url when provider changes
            def on_provider_change(provider):
                config = get_provider_config(provider)
                models = refresh_models_list()
                return models, gr.update(value=config["base_url"])

            provider_input.change(
                fn=on_provider_change,
                inputs=[provider_input],
                outputs=[model_input_settings, base_url_input]
            )
            refresh_models_settings_btn.click(fn=refresh_models_list, outputs=[model_input_settings])

            batch_size_settings = gr.Number(value=state.batch_size, label="Batch Size", precision=0)

            seed_input = gr.Textbox(value=str(state.seed or ""), label="Generation Seed (empty = random)")

            save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
            settings_status = gr.Markdown()

            save_settings_btn.click(
                fn=update_settings,
                inputs=[provider_input, model_input_settings, base_url_input, seed_input, batch_size_settings],
                outputs=[settings_status]
            )

            # OpenRouter Model Browser
            gr.Markdown("---")
            with gr.Accordion("🌐 OpenRouter Model Browser", open=False):
                gr.Markdown("*Browse OpenRouter models with pricing. Input/Output prices are per 1K tokens.*")
                with gr.Row():
                    fetch_or_btn = gr.Button("🔄 Fetch OpenRouter Models", scale=2)
                    or_model_selector = gr.Dropdown(
                        label="Quick Select",
                        choices=[],
                        allow_custom_value=True,
                        scale=3
                    )
                    use_or_model_btn = gr.Button("Use Selected", scale=1)

                or_models_display = gr.HTML("")

                fetch_or_btn.click(
                    fn=fetch_openrouter_models,
                    outputs=[or_models_display, or_model_selector]
                )

                def apply_or_model(model_id):
                    if model_id:
                        return gr.update(value=model_id), gr.update(value="openrouter")
                    return gr.update(), gr.update()

                use_or_model_btn.click(
                    fn=apply_or_model,
                    inputs=[or_model_selector],
                    outputs=[model_input_settings, provider_input]
                )

            gr.Markdown("---")
            gr.Markdown("### Run Management")
            with gr.Row():
                manage_run_selector = gr.Dropdown(
                    label="Select Run to Manage",
                    choices=get_available_runs(include_new=False),
                    interactive=True
                )
                refresh_manage_btn = gr.Button("🔄", scale=0)

            with gr.Row():
                new_run_name = gr.Textbox(label="New Name", placeholder="Enter new name")
                rename_btn = gr.Button("Rename Run")
                delete_btn = gr.Button("Delete Run", variant="stop")

            manage_status = gr.Markdown()

            refresh_manage_btn.click(
                fn=lambda: gr.update(choices=get_available_runs(include_new=False)),
                outputs=[manage_run_selector]
            )

            rename_btn.click(
                fn=rename_run_action,
                inputs=[manage_run_selector, new_run_name],
                outputs=[manage_status, manage_run_selector]
            )

            delete_btn.click(
                fn=delete_run_action,
                inputs=[manage_run_selector],
                outputs=[manage_status, train_run_selector, manage_run_selector]
            )

            gr.Markdown("---")
            gr.Markdown("### Export Dataset")
            with gr.Row():
                export_run_selector = gr.Dropdown(
                    label="Select Run to Export",
                    choices=get_available_runs(include_new=False),
                    interactive=True
                )
                export_btn = gr.Button("Export Samples to JSONL", variant="secondary")
            export_status = gr.Markdown()

            export_btn.click(
                fn=export_dataset,
                outputs=[export_status]
            )

            gr.Markdown("""

            ---
            **Provider Notes:**
            - **ollama**: Local, free. Run `ollama serve` first.
            - **openrouter**: Cloud. Needs `OPENROUTER_API_KEY` in `.env`
            - **lmstudio**: Local. Start server in LM Studio.
            - **openai**: Cloud. Needs `OPENAI_API_KEY` in `.env`
            """)

    gr.Markdown("""
    ---
    **Ethical AI Core** | [Core Principles](./core_principles.md)
    """)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

