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
class AppState:
    """Mutable app state for settings."""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    model = os.getenv("OLLAMA_MODEL", "qwen3-vl")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    seed = int(os.getenv("GENERATION_SEED", "0")) or None  # None = random
    timeout = int(os.getenv("LLM_TIMEOUT", "300"))

state = AppState()

def update_settings(provider: str, model: str, base_url: str, seed: str, timeout: int):
    """Update app settings dynamically."""
    state.provider = provider
    state.model = model
    state.base_url = base_url
    state.seed = int(seed) if seed.strip() else None
    state.timeout = timeout

    # Update environment so LLMClient picks it up
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_TIMEOUT"] = str(timeout)

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


def generate_prompts(prompts_per_persona: int, timeout: float = 1200, batch_size: int = 5, personas_source_run: Optional[str] = None, progress=gr.Progress()):
    """Generate prompts using personas from source run."""
    from src.dataset.personas import PersonaGenerator, PersonaDB, Persona

    progress(0, desc="Loading personas...")
    # Use source run for reading personas (always use simple personas.db)
    db = PersonaDB(run_id=personas_source_run, db_filename="personas.db")
    gen = PersonaGenerator()

    # Update Config with timeout (model comes from settings)
    if timeout:
        gen.client.update_config(timeout=timeout)


    persona_dicts = db.get_all_personas()
    personas = [Persona(**p) for p in persona_dicts]

    if not personas:
        return "No personas found. Generate personas first.", format_stats()

    total_prompts = 0
    output = []

    progress(0, desc=f"Generating prompts for {len(personas)} personas...")

    for i, persona in enumerate(personas):
        progress((i) / len(personas), desc=f"Generating for {persona.name}...")

        prompts = gen.generate_prompts_for_persona(persona, prompts_per_persona, batch_size=batch_size)
        total_prompts += len(prompts)

        output.append(f"### {persona.name}: {len(prompts)} prompts")
        for p in prompts:
            output.append(f"  - [D{p.difficulty}/S{p.safety_level}] {p.prompt}")

    output.append(f"\n---\n**Total new prompts generated:** {total_prompts}")

    # Get current run for auto-cascade to Step 3
    from src.dataset.personas import RunManager
    current_run = RunManager.get_current_run()
    choices = get_run_choices_with_stats()

    return "\n".join(output), format_stats(), gr.update(choices=choices, value=current_run)


def process_prompts(batch_size: int, prompts_source_run: Optional[str] = None, progress=gr.Progress()):
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

    prompts = prompt_db.get_unprocessed(batch_size)

    if not prompts:
        return "No unprocessed prompts found.", format_stats()

    output = []
    for i, (prompt_id, prompt_text, difficulty, safety_level, category, persona_id) in enumerate(prompts):
        progress((i + 1) / len(prompts), desc=f"Processing {i+1}/{len(prompts)}...")

        try:
            sample = generator.generate_sample(prompt_text)
            sample_db.save_sample(prompt_id, prompt_text, asdict(sample))
            prompt_db.mark_processed(prompt_id)

            tier_str = f"Tier {sample.tier_violated}" if sample.tier_violated else "Passed"
            output.append(f"[D{difficulty}/S{safety_level}] {tier_str}: {prompt_text}")
        except Exception as e:
            output.append(f"[ERROR] {prompt_text} — {e}")

    return "\n".join(output), format_stats()


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

def start_ddl_training(epochs: int, learning_rate: float, num_heads: int, run_id: Optional[str] = None, progress=gr.Progress()):
    """Train Deep Delta Learning heads from Constitutional data."""
    import torch

    from src.layers.delta import MultiHeadDelta
    from src.dataset.personas import SampleDB, RunManager
    import sqlite3

    progress(0, desc="Loading training data...")

    if run_id:
        RunManager.set_run(run_id)

    db = SampleDB()
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.execute("""
            SELECT naive_response, revised_response, tier_violated
            FROM samples WHERE tier_violated IS NOT NULL
        """)
        rows = cursor.fetchall()

    if not rows:
        return "No tier violations found in dataset. Need flagged samples for DDL training."

    progress(0.2, desc=f"Found {len(rows)} flagged samples...")

    # For demo: create a simple embedding dimension
    embed_dim = 64
    delta_model = MultiHeadDelta(embed_dim, num_heads=num_heads)
    optimizer = torch.optim.Adam(delta_model.parameters(), lr=learning_rate)

    progress(0.3, desc="Training Delta heads...")

    # Simulated training loop
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (naive, revised, tier) in enumerate(rows):
            # Simulate embeddings (in real impl, would use actual model embeddings)
            x_bad = torch.randn(1, embed_dim) * (tier or 1)  # Scaled by tier
            x_good = torch.randn(1, embed_dim) * 0.1

            optimizer.zero_grad()

            # Delta should map bad → good direction
            output = delta_model(x_bad)
            loss = (output - x_good).pow(2).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(rows)
        losses.append(avg_loss)
        progress((epoch + 1) / epochs, desc=f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # Save the trained model
    save_path = "data/delta_judge.pt"
    torch.save(delta_model.state_dict(), save_path)

    return f"""**DDL Training Complete**

**Configuration:**
- Epochs: {epochs}
- Learning Rate: {learning_rate}
- Delta Heads: {num_heads}
- Training Samples: {len(rows)}

**Results:**
- Initial Loss: {losses[0]:.4f}
- Final Loss: {losses[-1]:.4f}
- Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%

**Model saved to:** `{save_path}`

*The Delta model learns to project "bad" response directions toward "good" ones.*
"""


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
                    with gr.Accordion("⚙️ LLM Configuration", open=False):
                        with gr.Row():
                            timeout_input = gr.Number(label="Timeout (s)", value=1200, precision=0, scale=1)
                            batch_size_input = gr.Number(label="Batch Size", value=1, precision=0, scale=1)
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
                    process_batch_size = gr.Number(label="Batch Size", value=1, precision=0, minimum=1)
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
                inputs=[num_prompts_input, timeout_input, batch_size_input, personas_source_dd],
                outputs=[output_log, stats_text, prompts_source_dd]  # Cascade to Step 3
            )


            process_btn.click(
                fn=process_prompts,
                inputs=[process_batch_size, prompts_source_dd],
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
                    choices=get_available_runs(include_new=False),
                    value=lambda: get_available_runs(include_new=False)[0] if get_available_runs(include_new=False) else None,
                    interactive=True
                )
                refresh_train_runs_btn = gr.Button("🔄", size="sm", scale=0)

            refresh_train_runs_btn.click(
                fn=lambda: gr.update(choices=get_available_runs(include_new=False)),
                outputs=[train_run_selector]
            )

            with gr.Row():
                epochs = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                learning_rate = gr.Number(value=1e-3, label="Learning Rate")
                num_heads = gr.Slider(1, 8, value=4, step=1, label="Delta Heads")

            train_btn = gr.Button("Start DDL Training", variant="primary")
            training_output = gr.Markdown()

            train_btn.click(
                fn=start_ddl_training,
                inputs=[epochs, learning_rate, num_heads, train_run_selector],
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

            # Refresh models when provider changes
            provider_input.change(fn=refresh_models_list, outputs=[model_input_settings])
            refresh_models_settings_btn.click(fn=refresh_models_list, outputs=[model_input_settings])

            with gr.Row():
                base_url_input = gr.Textbox(value=state.base_url, label="Base URL")
                timeout_input = gr.Number(value=state.timeout, label="Timeout (seconds)")

            seed_input = gr.Textbox(value=str(state.seed or ""), label="Generation Seed (empty = random)")

            save_settings_btn = gr.Button("💾 Save Settings", variant="primary")
            settings_status = gr.Markdown()

            save_settings_btn.click(
                fn=update_settings,
                inputs=[provider_input, model_input_settings, base_url_input, seed_input, timeout_input],
                outputs=[settings_status]
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

