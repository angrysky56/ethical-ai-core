"""
Persona-Based Prompt Generation System

Pipeline:
1. Generate N simulated user personas (diverse interests, backgrounds)
2. For each persona, generate M prompts across difficulty/safety gradients
3. Store in SQLite database (scales better than JSONL)
4. Feed prompts to Constitutional pipeline
"""

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import GLOBAL_SEED, PROJECT_ROOT, get_run_id
from src.llm_client import get_llm_client


@dataclass
class Persona:
    id: str
    name: str
    age: str
    occupation: str
    interests: list[str]
    expertise_level: str  # novice, intermediate, expert
    communication_style: str
    background: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GeneratedPrompt:
    id: str
    persona_id: str
    prompt: str
    difficulty: int  # 1-5
    safety_level: int  # 1-5 (1=safe, 5=adversarial)
    category: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    processed: bool = False


from src.dataset.loader import get_loader


def get_detailed_template():
    return get_loader().load_personas_config().get("detailed_template", "")


def get_simple_template():
    return get_loader().load_personas_config().get("simple_template", "")


def get_prompt_generation_template():
    return get_loader().load_personas_config().get("prompt_generation_template", "")


class RunManager:
    """Manages per-run database directories."""

    _current_run_id: Optional[str] = None
    _run_dir: Optional[Path] = None

    @classmethod
    def get_runs_root(cls, category: str = "default") -> Path:
        """Get root directory for specific run category."""
        base = PROJECT_ROOT / "data" / "runs"
        if category == "detailed":
            return base / "detailed"
        return base

    @classmethod
    def resolve_run_path(cls, run_id: str) -> Path:
        """Find path for a run ID by checking locations."""
        # Check detailed first
        detailed = cls.get_runs_root("detailed") / run_id
        if detailed.exists():
            return detailed

        # Check default
        default = cls.get_runs_root("default") / run_id
        return default

    @classmethod
    def get_current_run(cls) -> str:
        """Get or create current run ID."""
        if cls._current_run_id is None:
            cls._current_run_id = get_run_id()
        return cls._current_run_id

    @classmethod
    def get_run_dir(cls) -> Path:
        """Get current run directory."""
        if cls._run_dir is None:
            # If we don't know where it is, assume default for new ones?
            # Or use resolve if ID is set
            run_id = cls.get_current_run()
            cls._run_dir = cls.resolve_run_path(run_id)
            cls._run_dir.mkdir(parents=True, exist_ok=True)
        return cls._run_dir

    @classmethod
    def set_run(cls, run_id: str):
        """Switch to a different run."""
        cls._current_run_id = run_id
        cls._run_dir = cls.resolve_run_path(run_id)
        # Don't mkdir here, wait for usage, or check existence?
        # Actually set_run usually implies loading existing.

    @classmethod
    def new_run(cls, category: str = "default") -> str:
        """Create and switch to a new run in specified category."""
        new_id = get_run_id()
        cls._current_run_id = new_id
        root = cls.get_runs_root(category)
        cls._run_dir = root / new_id

        cls._run_dir.mkdir(parents=True, exist_ok=True)
        return new_id

    @classmethod
    def delete_run(cls, run_id: str) -> bool:
        """Delete a run directory."""
        import shutil

        run_path = cls.resolve_run_path(run_id)
        if run_path.exists() and run_path.is_dir():
            shutil.rmtree(run_path)
            if cls._current_run_id == run_id:
                cls._current_run_id = None
                cls._run_dir = None
            return True
        return False

    @classmethod
    def list_runs(cls, category: str = "default") -> list[str]:
        """List all available runs in a category."""
        runs_dir = cls.get_runs_root(category)
        if not runs_dir.exists():
            return []
        return sorted([d.name for d in runs_dir.iterdir() if d.is_dir()], reverse=True)


class PersonaDB:
    """SQLite database for personas."""

    def __init__(self, run_id: Optional[str] = None, db_filename: str = "personas.db"):
        if run_id:
            self.run_dir = RunManager.resolve_run_path(run_id)
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / db_filename

        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    age TEXT,
                    occupation TEXT,
                    interests TEXT,  -- JSON array
                    expertise_level TEXT,
                    communication_style TEXT,
                    background TEXT,
                    created_at TEXT
                )
            """
            )
            conn.commit()

    def save_persona(self, persona: Persona):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO personas
                (id, name, age, occupation, interests, expertise_level,
                 communication_style, background, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    persona.id,
                    persona.name,
                    persona.age,
                    persona.occupation,
                    json.dumps(persona.interests),
                    persona.expertise_level,
                    persona.communication_style,
                    persona.background,
                    persona.created_at,
                ),
            )
            conn.commit()

    def get_all_personas(self) -> list[dict]:
        """Get all personas from database."""
        with sqlite3.connect(self.db_path) as conn:
            # Check if age column exists (for backward compatibility)
            cursor = conn.execute("PRAGMA table_info(personas)")
            columns = [info[1] for info in cursor.fetchall()]

            if "age" in columns:
                cursor = conn.execute(
                    """
                    SELECT id, name, age, occupation, interests,
                           expertise_level, communication_style, background, created_at
                    FROM personas ORDER BY created_at DESC
                """
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, name, age_range, occupation, interests,
                           expertise_level, communication_style, background, created_at
                    FROM personas ORDER BY created_at DESC
                """
                )
            rows = cursor.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "age": r[2],
                "occupation": r[3],
                "interests": json.loads(r[4]),
                "expertise_level": r[5],
                "communication_style": r[6],
                "background": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM personas").fetchone()[0]


class PromptDB:
    """SQLite database for prompts/questions."""

    def __init__(self, run_id: Optional[str] = None):
        if run_id:
            self.run_dir = RunManager.resolve_run_path(run_id)
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / "prompts.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT,
                    prompt TEXT,
                    difficulty INTEGER,
                    safety_level INTEGER,
                    category TEXT,
                    created_at TEXT,
                    processed INTEGER DEFAULT 0
                )
            """
            )
            conn.commit()

    def save_prompt(self, prompt: GeneratedPrompt):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO prompts
                (id, persona_id, prompt, difficulty, safety_level, category, created_at, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prompt.id,
                    prompt.persona_id,
                    prompt.prompt,
                    prompt.difficulty,
                    prompt.safety_level,
                    prompt.category,
                    prompt.created_at,
                    prompt.processed,
                ),
            )
            conn.commit()

    def get_unprocessed(self, limit: int = None) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            if limit:
                cursor = conn.execute(
                    """
                    SELECT id, prompt, difficulty, safety_level, category, persona_id
                    FROM prompts WHERE processed = 0
                    ORDER BY safety_level ASC, difficulty ASC
                    LIMIT ?
                """,
                    (limit,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, prompt, difficulty, safety_level, category, persona_id
                    FROM prompts WHERE processed = 0
                    ORDER BY safety_level ASC, difficulty ASC
                """
                )
            return cursor.fetchall()

    def mark_processed(self, prompt_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE prompts SET processed = 1 WHERE id = ?", (prompt_id,))
            conn.commit()

    def get_all(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, prompt, difficulty, safety_level, category, processed, created_at
                FROM prompts ORDER BY created_at DESC LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()

        return [
            {
                "id": r[0],
                "prompt": r[1],
                "difficulty": r[2],
                "safety_level": r[3],
                "category": r[4],
                "processed": bool(r[5]),
                "created_at": r[6],
            }
            for r in rows
        ]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]

    def count_unprocessed(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM prompts WHERE processed = 0"
            ).fetchone()[0]


class SampleDB:
    """SQLite database for constitutional samples (dataset)."""

    def __init__(self, run_id: Optional[str] = None):
        if run_id:
            self.run_dir = RunManager.resolve_run_path(run_id)
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / "samples.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY,
                    prompt_id TEXT,
                    prompt_text TEXT,
                    naive_response TEXT,
                    critique TEXT,
                    revised_response TEXT,
                    principle_attribution TEXT,
                    tier_violated INTEGER,
                    created_at TEXT
                )
            """
            )
            conn.commit()

    def save_sample(self, prompt_id: str, prompt_text: str, sample: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO samples
                (id, prompt_id, prompt_text, naive_response, critique, revised_response,
                 principle_attribution, tier_violated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    sample["sample_id"],
                    prompt_id,
                    prompt_text,
                    sample["naive_response"],
                    sample["critique"],
                    sample["revised_response"],
                    sample["principle_attribution"],
                    sample["tier_violated"],
                    sample["timestamp"],
                ),
            )
            conn.commit()

    def get_all(self, limit: int = None) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            if limit:
                cursor = conn.execute(
                    """
                    SELECT id, prompt_id, prompt_text, naive_response, critique,
                           revised_response, principle_attribution, tier_violated, created_at
                    FROM samples ORDER BY created_at DESC LIMIT ?
                """,
                    (limit,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, prompt_id, prompt_text, naive_response, critique,
                           revised_response, principle_attribution, tier_violated, created_at
                    FROM samples ORDER BY created_at DESC
                """
                )
            rows = cursor.fetchall()

        return [
            {
                "id": r[0],
                "prompt_id": r[1],
                "prompt_text": r[2],
                "naive_response": r[3],
                "critique": r[4],
                "revised_response": r[5],
                "principle_attribution": r[6],
                "tier_violated": r[7],
                "created_at": r[8],
            }
            for r in rows
        ]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]

    def export_jsonl(self, output_path: Path) -> int:
        """Export samples to JSONL format."""
        samples = self.get_all(limit=100000)
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return len(samples)


def get_run_stats() -> dict:
    """Get stats for current run."""
    run_id = RunManager.get_current_run()
    persona_db = PersonaDB()
    prompt_db = PromptDB()
    sample_db = SampleDB()

    return {
        "run_id": run_id,
        "run_dir": str(RunManager.get_run_dir()),
        "personas": persona_db.count(),
        "prompts": prompt_db.count(),
        "unprocessed": prompt_db.count_unprocessed(),
        "samples": sample_db.count(),
    }


class PersonaGenerator:
    """Generates personas and prompts using local LLM."""

    def __init__(self, db_filename: str = "personas.db"):
        self.client = get_llm_client()
        self.persona_db = PersonaDB(db_filename=db_filename)
        self.prompt_db = PromptDB()

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

    def _parse_json_array(self, text: str) -> list:
        """Extract JSON array or single object from LLM response."""
        import re

        # Find JSON array in code block
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Try to find JSON object in code block (single item)
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # Try to find array directly
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end > start:
                    text = text[start:end]
                else:
                    # Try to find single object directly
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end > start:
                        text = text[start:end]

        try:
            result = json.loads(text)
            # If it's a single dict, wrap in a list
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result
            else:
                print(f"[ERROR] Unexpected JSON type: {type(result)}")
                return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON response: {e}")
            print(f"[DEBUG] Partial Text: {text[:200]}...")
            return []

    def generate_personas(
        self, count: int = 10, seed: Optional[int] = None, mode: str = "simple"
    ) -> list[Persona]:
        """Generate diverse user personas with optional seed for variation."""
        if seed is None:
            seed = GLOBAL_SEED

        template = (
            get_simple_template() if mode == "simple" else get_detailed_template()
        )
        print(f"Generating {count} personas (seed={seed}, mode={mode})...")

        prompt = template.format(count=count, seed=seed)
        response = self.client.complete(
            prompt,
            system="You are a data generator. You MUST respond with a valid, raw JSON array only. No markdown. No conversational text.",
            # Don't specify temperature - let model use its optimal default
        )

        print("-" * 40)
        print(f"LLM Response:\n{response}")
        print("-" * 40)

        personas = []
        for p in self._parse_json_array(response):
            persona = Persona(
                id=self._generate_id(p.get("name", "")),
                name=p.get("name", "Unknown"),
                age=str(p.get("age", p.get("age_range", "unknown"))),
                occupation=p.get("occupation", "any"),
                interests=p.get("interests", []),
                expertise_level=p.get("expertise_level", "any"),
                communication_style=p.get("communication_style", "any"),
                background=p.get("background", ""),
            )
            self.persona_db.save_persona(persona)
            personas.append(persona)
            print(
                f"  ✓ Saved Persona:\n     Name: {persona.name}\n     Role: {persona.occupation}\n     Bio: {persona.background}"
            )

        return personas

    def generate_prompts_for_persona(
        self,
        persona: Persona,
        count: int = 10,
        batch_size: int = 5,
        min_difficulty: int = 1,
        min_safety: int = 1,
    ) -> list[GeneratedPrompt]:
        """Generate prompts for a specific persona across difficulty/safety gradient."""
        print(
            f"Generating {count} prompts for {persona.name} (batch_size={batch_size}, min_d={min_difficulty}, min_s={min_safety})..."
        )

        all_prompts = []
        remaining = count

        while remaining > 0:
            current_batch = min(remaining, batch_size)
            print(f"  > Batch request: {current_batch} prompts...")

            # Prepare template context with robust age handling
            context = {
                "name": persona.name,
                "age": persona.age,
                "age_range": persona.age,  # Backwards compat
                "occupation": persona.occupation,
                "interests": ", ".join(persona.interests),
                "expertise_level": persona.expertise_level,
                "communication_style": persona.communication_style,
                "background": persona.background,
                "count": current_batch,
                "min_difficulty": min_difficulty,
                "min_safety": min_safety,
            }

            prompt = get_prompt_generation_template().format(**context)

            try:
                response = self.client.complete(
                    prompt,
                    system="You are simulating a user persona. Generate realistic prompts they would ask. Include edge cases and ethical dilemmas. Always respond with valid JSON.",
                    # Don't specify temperature - let model use its optimal default
                )
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                remaining -= current_batch  # Skip this batch to avoid infinite loop
                continue

            print("-" * 40)
            print(f"Generated Prompts for {persona.name} (Batch):\n{response}")
            print("-" * 40)

            batch_prompts = []
            for p in self._parse_json_array(response):
                gen_prompt = GeneratedPrompt(
                    id=self._generate_id(p.get("prompt", "")),
                    persona_id=persona.id,
                    prompt=p.get("prompt", ""),
                    difficulty=p.get("difficulty", 3),
                    safety_level=p.get("safety_level", 1),
                    category=p.get("category", "general"),
                )
                self.prompt_db.save_prompt(gen_prompt)
                batch_prompts.append(gen_prompt)
                all_prompts.append(gen_prompt)
                print(
                    f"  ✓ [{gen_prompt.category}] D{gen_prompt.difficulty}/S{gen_prompt.safety_level}: {gen_prompt.prompt}"
                )

            remaining -= current_batch
            if not batch_prompts:
                print("[WARN] No valid prompts parsed from this batch.")

        print(f"  ✓ Saved Total {len(all_prompts)} prompts to DB")
        return all_prompts

    def inject_benchmark_prompts(self) -> int:
        """Inject static benchmark prompts from the active pack into the DB."""
        from src.dataset.prompts import get_prompts_by_category

        prompts_map = get_prompts_by_category()
        count = 0

        # Use a consistent ID for the 'benchmark' pseudo-persona
        benchmark_persona_id = "BENCHMARK_SET"

        for category, prompts in prompts_map.items():
            if not isinstance(prompts, list):
                continue

            for p_text in prompts:
                gen_prompt = GeneratedPrompt(
                    id=self._generate_id(p_text),
                    persona_id=benchmark_persona_id,
                    prompt=p_text,
                    difficulty=3,  # Default
                    safety_level=5 if category == "deontology" else 3,
                    category=category,
                    processed=False,
                )
                self.prompt_db.save_prompt(gen_prompt)
                count += 1
                print(f"  ✓ Injected Benchmark: [{category}] {p_text[:50]}...")

        return count

    def run_pipeline(self, num_personas: int = 5, prompts_per_persona: int = 10):
        """Full pipeline: generate personas → prompts → save to DB."""
        print(f"\n{'='*60}")
        print("PERSONA-BASED PROMPT GENERATION PIPELINE")
        print(f"{'='*60}")

        # Step 1: Generate personas
        print(f"\n[1/2] Generating {num_personas} personas...")
        personas = self.generate_personas(num_personas)

        # Step 2: Generate prompts for each persona
        print("\n[2/2] Generating prompts for each persona...")
        total_prompts = 0
        for persona in personas:
            prompts = self.generate_prompts_for_persona(persona, prompts_per_persona)
            total_prompts += len(prompts)

        print(f"\n{'='*60}")
        print(f"✓ Generated {len(personas)} personas, {total_prompts} prompts")
        print(f"✓ Saved to: {RunManager.get_run_dir()}")
        print(f"{'='*60}")

        return get_run_stats()


def process_unprocessed_prompts(limit: int = 10):
    """Process unprocessed prompts through Constitutional pipeline."""
    from src.dataset.generator import ConstitutionalGenerator

    prompt_db = PromptDB()
    sample_db = SampleDB()
    generator = ConstitutionalGenerator()

    prompts = prompt_db.get_unprocessed(limit)
    print(f"Processing {len(prompts)} unprocessed prompts...")

    for prompt_id, prompt_text, difficulty, safety_level, category, _ in prompts:
        print(f"\n[{category}] [D{difficulty}/S{safety_level}] {prompt_text}")
        try:
            sample = generator.generate_sample(prompt_text)
            sample_db.save_sample(prompt_id, prompt_text, asdict(sample))
            prompt_db.mark_processed(prompt_id)
            print(f"  ✓ Tier violated: {sample.tier_violated}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    return {
        "prompts": prompt_db.count(),
        "unprocessed": prompt_db.count_unprocessed(),
        "samples": sample_db.count(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persona-based prompt generation")
    parser.add_argument(
        "--personas", type=int, default=5, help="Number of personas to generate"
    )
    parser.add_argument("--prompts", type=int, default=10, help="Prompts per persona")
    parser.add_argument(
        "--process", type=int, metavar="N", help="Process N unprocessed prompts"
    )
    parser.add_argument("--stats", action="store_true", help="Show database stats")

    # Optional run selection
    parser.add_argument("--run", type=str, help="Specific run ID to use")

    args = parser.parse_args()

    if args.run:
        RunManager.set_run(args.run)

    if args.stats:
        print(json.dumps(get_run_stats(), indent=2))
    elif args.process:
        stats = process_unprocessed_prompts(args.process)
        print(json.dumps(stats, indent=2))
    else:
        gen = PersonaGenerator()
        stats = gen.run_pipeline(args.personas, args.prompts)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
