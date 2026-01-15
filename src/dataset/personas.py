"""
Persona-Based Prompt Generation System

Pipeline:
1. Generate N simulated user personas (diverse interests, backgrounds)
2. For each persona, generate M prompts across difficulty/safety gradients
3. Store in SQLite database (scales better than JSONL)
4. Feed prompts to Constitutional pipeline
"""
import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
from src.llm_client import get_llm_client
from src.config import PROJECT_ROOT, GLOBAL_SEED, get_run_id


@dataclass
class Persona:
    id: str
    name: str
    age_range: str
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


PERSONA_GENERATION_TEMPLATE = """Generate {count} diverse simulated user personas. Each persona should have unique characteristics.

Seed for variation: {seed}

Include a wide range of:
- Ages (students to retirees)
- Occupations (tech, healthcare, arts, trades, academics, etc.)
- Expertise levels (novice to expert)
- Communication styles (formal, casual, technical, creative)
- Cultural/regional backgrounds

For each persona, provide:
- Name (realistic, diverse)
- Age range (e.g., "25-30")
- Occupation
- 3-5 interests
- Expertise level in their field
- Communication style
- Brief background (1-2 sentences)

CRITICAL INSTRUCTION: Respond with a RAW JSON ARRAY only. Do not wrap in markdown code blocks. Do not include any text before or after the JSON.

Example Format:
[
  {{
    "name": "Name Here",
    "age_range": "30-40",
    "occupation": "Job Title",
    "interests": ["Interest 1", "Interest 2"],
    "expertise_level": "intermediate",
    "communication_style": "formal",
    "background": "Short bio."
  }}
]
"""


SIMPLE_PERSONA_TEMPLATE = """Generate {count} functional user personas.
Focus on filling the required fields concisely, suitable for generating technical questions.
Do not elaborate unnecessarily.

Fields:
- Name
- Age range
- Occupation
- Interests (list)
- Expertise level (novice|intermediate|expert)
- Communication style
- Brief background

Respond with a JSON array:
```json
[
  {{
    "name": "...",
    "age_range": "...",
    "occupation": "...",
    "interests": ["...", "..."],
    "expertise_level": "...",
    "communication_style": "...",
    "background": "..."
  }}
]
```"""

PROMPT_GENERATION_TEMPLATE = """You are {name}, a {age_range} year old {occupation}.
Your interests include: {interests}.
Your expertise level is {expertise_level} and you communicate in a {communication_style} way.
Background: {background}

Generate {count} questions/prompts that someone like you would ask an AI assistant.

Create prompts across a GRADIENT of:
- **Difficulty**: 1 to {min_difficulty} (1=simple, 5=complex/technical)
- **Danger Level**: 1 to {min_safety} (1=safe, 5=ethically challenging/adversarial)

Focus on:
- Questions with difficulty up to {min_difficulty}
- Questions with danger level up to {min_safety}
- Edge cases and dilemmas when max values are 3+
- Adversarial/red-team requests that test ethical boundaries when danger is 4-5

Respond with a JSON array:
```json
[
  {{
    "prompt": "...",
    "difficulty": 1-{min_difficulty},
    "safety_level": 1-{min_safety},
    "category": "general|technical|ethical_dilemma|adversarial"
  }}
]
```"""


class RunManager:
    """Manages per-run database directories."""
    _current_run_id: Optional[str] = None
    _run_dir: Optional[Path] = None

    @classmethod
    def get_current_run(cls) -> str:
        """Get or create current run ID."""
        if cls._current_run_id is None:
            cls._current_run_id = get_run_id()
        return cls._current_run_id

    @classmethod
    def get_run_dir(cls) -> Path:
        """Get current run directory, creating if needed."""
        if cls._run_dir is None:
            cls._run_dir = PROJECT_ROOT / "data" / "runs" / cls.get_current_run()
            cls._run_dir.mkdir(parents=True, exist_ok=True)
        return cls._run_dir

    @classmethod
    def set_run(cls, run_id: str):
        """Switch to a different run."""
        cls._current_run_id = run_id
        cls._run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        cls._run_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new_run(cls) -> str:
        """Create and switch to a new run."""
        new_id = get_run_id()
        cls.set_run(new_id)
        return new_id

    @classmethod
    def delete_run(cls, run_id: str) -> bool:
        """Delete a run directory. Returns True if successful."""
        import shutil
        run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir)
            if cls._current_run_id == run_id:
                cls._current_run_id = None
                cls._run_dir = None
            return True
        return False

    @classmethod
    def rename_run(cls, run_id: str, new_name: str) -> bool:
        """Rename a run directory. Returns True if successful."""
        # Sanitize new name
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', new_name)

        runs_dir = PROJECT_ROOT / "data" / "runs"
        old_path = runs_dir / run_id
        new_path = runs_dir / clean_name

        if not old_path.exists():
            return False
        if new_path.exists():
            return False

        old_path.rename(new_path)

        if cls._current_run_id == run_id:
            cls._current_run_id = clean_name
            cls._run_dir = new_path

        return True

    @classmethod
    def list_runs(cls) -> list[str]:
        """List all available runs."""
        runs_dir = PROJECT_ROOT / "data" / "runs"
        if not runs_dir.exists():
            return []
        return sorted([d.name for d in runs_dir.iterdir() if d.is_dir()], reverse=True)


class PersonaDB:
    """SQLite database for personas."""

    def __init__(self, run_id: Optional[str] = None, db_filename: str = "personas.db"):
        if run_id:
            self.run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / db_filename
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    age_range TEXT,
                    occupation TEXT,
                    interests TEXT,  -- JSON array
                    expertise_level TEXT,
                    communication_style TEXT,
                    background TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def save_persona(self, persona: Persona):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO personas
                (id, name, age_range, occupation, interests, expertise_level,
                 communication_style, background, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                persona.id, persona.name, persona.age_range, persona.occupation,
                json.dumps(persona.interests), persona.expertise_level,
                persona.communication_style, persona.background, persona.created_at
            ))
            conn.commit()

    def get_all_personas(self) -> list[dict]:
        """Get all personas from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, age_range, occupation, interests,
                       expertise_level, communication_style, background, created_at
                FROM personas ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()

        return [
            {
                "id": r[0], "name": r[1], "age_range": r[2], "occupation": r[3],
                "interests": json.loads(r[4]), "expertise_level": r[5],
                "communication_style": r[6], "background": r[7], "created_at": r[8]
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
            self.run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / "prompts.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            """)
            conn.commit()

    def save_prompt(self, prompt: GeneratedPrompt):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO prompts
                (id, persona_id, prompt, difficulty, safety_level, category, created_at, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt.id, prompt.persona_id, prompt.prompt, prompt.difficulty,
                prompt.safety_level, prompt.category, prompt.created_at, prompt.processed
            ))
            conn.commit()

    def get_unprocessed(self, limit: int = None) -> list[tuple]:
        with sqlite3.connect(self.db_path) as conn:
            if limit:
                cursor = conn.execute("""
                    SELECT id, prompt, difficulty, safety_level, category, persona_id
                    FROM prompts WHERE processed = 0
                    ORDER BY safety_level ASC, difficulty ASC
                    LIMIT ?
                """, (limit,))
            else:
                cursor = conn.execute("""
                    SELECT id, prompt, difficulty, safety_level, category, persona_id
                    FROM prompts WHERE processed = 0
                    ORDER BY safety_level ASC, difficulty ASC
                """)
            return cursor.fetchall()

    def mark_processed(self, prompt_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE prompts SET processed = 1 WHERE id = ?", (prompt_id,))
            conn.commit()

    def get_all(self, limit: int = 100) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, prompt, difficulty, safety_level, category, processed, created_at
                FROM prompts ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

        return [
            {"id": r[0], "prompt": r[1], "difficulty": r[2], "safety_level": r[3],
             "category": r[4], "processed": bool(r[5]), "created_at": r[6]}
            for r in rows
        ]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]

    def count_unprocessed(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM prompts WHERE processed = 0").fetchone()[0]


class SampleDB:
    """SQLite database for constitutional samples (dataset)."""

    def __init__(self, run_id: Optional[str] = None):
        if run_id:
            self.run_dir = PROJECT_ROOT / "data" / "runs" / run_id
        else:
            self.run_dir = RunManager.get_run_dir()
        self.db_path = self.run_dir / "samples.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            """)
            conn.commit()

    def save_sample(self, prompt_id: str, prompt_text: str, sample: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO samples
                (id, prompt_id, prompt_text, naive_response, critique, revised_response,
                 principle_attribution, tier_violated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sample['sample_id'], prompt_id, prompt_text, sample['naive_response'],
                sample['critique'], sample['revised_response'],
                sample['principle_attribution'], sample['tier_violated'],
                sample['timestamp']
            ))
            conn.commit()

    def get_all(self, limit: int = None) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            if limit:
                cursor = conn.execute("""
                    SELECT id, prompt_id, prompt_text, naive_response, critique,
                           revised_response, principle_attribution, tier_violated, created_at
                    FROM samples ORDER BY created_at DESC LIMIT ?
                """, (limit,))
            else:
                cursor = conn.execute("""
                    SELECT id, prompt_id, prompt_text, naive_response, critique,
                           revised_response, principle_attribution, tier_violated, created_at
                    FROM samples ORDER BY created_at DESC
                """)
            rows = cursor.fetchall()

        return [
            {"id": r[0], "prompt_id": r[1], "prompt_text": r[2], "naive_response": r[3],
             "critique": r[4], "revised_response": r[5], "principle_attribution": r[6],
             "tier_violated": r[7], "created_at": r[8]}
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
        "samples": sample_db.count()
    }


class PersonaGenerator:
    """Generates personas and prompts using local LLM."""

    def __init__(self, db_filename: str = "personas.db"):
        self.client = get_llm_client()
        self.persona_db = PersonaDB(db_filename=db_filename)
        self.prompt_db = PromptDB()

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:12]

    def _parse_json_array(self, text: str) -> list:
        """Extract JSON array or single object from LLM response."""
        import re

        # Find JSON array in code block
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Try to find JSON object in code block (single item)
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # Try to find array directly
                start = text.find('[')
                end = text.rfind(']') + 1
                if start != -1 and end > start:
                    text = text[start:end]
                else:
                    # Try to find single object directly
                    start = text.find('{')
                    end = text.rfind('}') + 1
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

    def generate_personas(self, count: int = 10, seed: Optional[int] = None, mode: str = "simple") -> list[Persona]:
        """Generate diverse user personas with optional seed for variation."""
        if seed is None:
            seed = GLOBAL_SEED

        template = SIMPLE_PERSONA_TEMPLATE if mode == "simple" else PERSONA_GENERATION_TEMPLATE
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
                id=self._generate_id(p.get('name', '')),
                name=p.get('name', 'Unknown'),
                age_range=p.get('age_range', '18-65'),
                occupation=p.get('occupation', 'any'),
                interests=p.get('interests', []),
                expertise_level=p.get('expertise_level', 'any'),
                communication_style=p.get('communication_style', 'any'),
                background=p.get('background', '')
            )
            self.persona_db.save_persona(persona)
            personas.append(persona)
            print(f"  ✓ Saved Persona:\n     Name: {persona.name}\n     Role: {persona.occupation}\n     Bio: {persona.background}")

        return personas

    def generate_prompts_for_persona(self, persona: Persona, count: int = 10, batch_size: int = 5, min_difficulty: int = 1, min_safety: int = 1) -> list[GeneratedPrompt]:
        """Generate prompts for a specific persona across difficulty/safety gradient."""
        print(f"Generating {count} prompts for {persona.name} (batch_size={batch_size}, min_d={min_difficulty}, min_s={min_safety})...")

        all_prompts = []
        remaining = count

        while remaining > 0:
            current_batch = min(remaining, batch_size)
            print(f"  > Batch request: {current_batch} prompts...")

            prompt = PROMPT_GENERATION_TEMPLATE.format(
                name=persona.name,
                age_range=persona.age_range,
                occupation=persona.occupation,
                interests=", ".join(persona.interests),
                expertise_level=persona.expertise_level,
                communication_style=persona.communication_style,
                background=persona.background,
                count=current_batch,
                min_difficulty=min_difficulty,
                min_safety=min_safety
            )

            try:
                response = self.client.complete(
                    prompt,
                    system="You are simulating a user persona. Generate realistic prompts they would ask. Include edge cases and ethical dilemmas. Always respond with valid JSON.",
                    # Don't specify temperature - let model use its optimal default
                )
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                remaining -= current_batch # Skip this batch to avoid infinite loop
                continue

            print("-" * 40)
            print(f"Generated Prompts for {persona.name} (Batch):\n{response}")
            print("-" * 40)

            batch_prompts = []
            for p in self._parse_json_array(response):
                gen_prompt = GeneratedPrompt(
                    id=self._generate_id(p.get('prompt', '')),
                    persona_id=persona.id,
                    prompt=p.get('prompt', ''),
                    difficulty=p.get('difficulty', 3),
                    safety_level=p.get('safety_level', 1),
                    category=p.get('category', 'general')
                )
                self.prompt_db.save_prompt(gen_prompt)
                batch_prompts.append(gen_prompt)
                all_prompts.append(gen_prompt)
                print(f"  ✓ [{gen_prompt.category}] D{gen_prompt.difficulty}/S{gen_prompt.safety_level}: {gen_prompt.prompt}")

            remaining -= current_batch
            if not batch_prompts:
                print("[WARN] No valid prompts parsed from this batch.")

        print(f"  ✓ Saved Total {len(all_prompts)} prompts to DB")
        return all_prompts

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

    for prompt_id, prompt_text, difficulty, safety_level, category in prompts:
        print(f"\n[D{difficulty}/S{safety_level}] {prompt_text}")
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
        "samples": sample_db.count()
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Persona-based prompt generation")
    parser.add_argument("--personas", type=int, default=5, help="Number of personas to generate")
    parser.add_argument("--prompts", type=int, default=10, help="Prompts per persona")
    parser.add_argument("--process", type=int, metavar="N", help="Process N unprocessed prompts")
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
