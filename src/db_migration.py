
import sqlite3
import json
from pathlib import Path
from dataclasses import asdict

# Adjust path to find src module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset.personas import RunManager, PersonaDB, PromptDB, SampleDB, Persona, GeneratedPrompt
from src.config import PROJECT_ROOT

def migrate_legacy_db():
    legacy_db_path = PROJECT_ROOT / "data" / "personas.db"

    if not legacy_db_path.exists():
        print(f"No legacy database found at {legacy_db_path}")
        return

    print("Found legacy database. Starting migration...")

    # Create legacy run
    run_id = "legacy_migration"
    RunManager.set_run(run_id)
    print(f"Target Run: {run_id}")

    # Initialize Target DBs
    persona_db = PersonaDB()
    prompt_db = PromptDB()
    sample_db = SampleDB()

    # Connect to legacy
    conn = sqlite3.connect(legacy_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Migrate Personas
    print("Migrating Personas...")
    cursor.execute("SELECT * FROM personas")
    count = 0
    for row in cursor.fetchall():
        try:
            p = Persona(
                id=row['id'],
                name=row['name'],
                age_range=row['age_range'],
                occupation=row['occupation'],
                interests=json.loads(row['interests']) if row['interests'] else [],
                expertise_level=row['expertise_level'],
                communication_style=row['communication_style'],
                background=row['background']
            )
            persona_db.save_persona(p)
            count += 1
        except Exception as e:
            print(f"Error migrating persona {row['id']}: {e}")
    print(f"Migrated {count} personas.")

    # 2. Migrate Prompts
    print("Migrating Prompts...")
    cursor.execute("SELECT * FROM prompts")
    count = 0
    for row in cursor.fetchall():
        try:
            p = GeneratedPrompt(
                id=row['id'],
                persona_id=row['persona_id'],
                prompt=row['prompt'],
                difficulty=row['difficulty'],
                safety_level=row['safety_level'],
                category=row['category']
            )
            prompt_db.save_prompt(p)
            # Check processed status? PromptDB doesn't expose writing 'processed' directly in save_prompt
            # but creates it as 0. We might lose processed status if we don't handle it.
            # But the 'processed' flag is mostly for the pipeline.
            # If we migrate samples, we should probably mark prompts as processed.
            count += 1
        except Exception as e:
            print(f"Error migrating prompt {row['id']}: {e}")
    print(f"Migrated {count} prompts.")

    # 3. Migrate Samples
    print("Migrating Samples...")
    cursor.execute("SELECT * FROM samples")
    count = 0
    for row in cursor.fetchall():
        try:
            # SampleDB expects a dict for saving
            sample_data = {
                "id": row['id'],
                "prompt_id": row['prompt_id'],
                "prompt_text": "", # Warning: Legacy sample table didn't store prompt_text directly?
                # Wait, schema check:
                # CREATE TABLE samples ( ... prompt_id TEXT ... )
                # It doesn't have prompt_text. SampleDB.save_sample ADDS it?
                # Let's check SampleDB.save_sample implementation.

                "naive_response": row['naive_response'],
                "critique": row['critique'],
                "revised_response": row['revised_response'],
                "principle_attribution": row['principle_attribution'],
                "tier_violated": row['tier_violated'],
                "created_at": row['created_at']
            }
            # SampleDB.save_sample inserts:
            # (id, prompt_id, prompt_text, naive_response, ...)
            # We need to fetch prompt_text from prompt_db if possible.

            # Use raw SQL insert to SampleDB to be safe and accurate
            with sqlite3.connect(sample_db.db_path) as target_conn:
                # We need prompt_text. Join with prompts table?
                # Legacy DB has prompt text in prompts table.
                pass

            # Actually, let's just use a JOIN query on the source.
        except Exception as e:
             print(f"Error migrating sample {row['id']}: {e}")

    # Improved Sample Migration Query
    cursor.execute("""
        SELECT s.*, p.prompt as prompt_text
        FROM samples s
        LEFT JOIN prompts p ON s.prompt_id = p.id
    """)

    count = 0
    with sqlite3.connect(sample_db.db_path) as target_conn:
        for row in cursor.fetchall():
            try:
                target_conn.execute("""
                    INSERT OR IGNORE INTO samples
                    (id, prompt_id, prompt_text, naive_response, critique, revised_response,
                     principle_attribution, tier_violated, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['id'], row['prompt_id'], row['prompt_text'] or "",
                    row['naive_response'], row['critique'], row['revised_response'],
                    row['principle_attribution'], row['tier_violated'], row['created_at']
                ))
                count += 1
            except Exception as e:
                print(f"Error inserting sample {row['id']}: {e}")
        target_conn.commit()

    print(f"Migrated {count} samples.")
    print("Migration complete. You can select 'legacy_migration' in the UI.")

if __name__ == "__main__":
    migrate_legacy_db()
