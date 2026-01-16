import os
import sys
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# ==========================================
# Configuration
# ==========================================
MAX_SEQ_LENGTH = 1024
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True

def train_local(
    data_path: str,
    output_dir: str,
    model_name: str = "unsloth/gemma-3-4b-it",
    epochs: int = 1,
    learning_rate: float = 2e-4,
):
    print(f"🚀 Starting Local Training")
    print(f"📂 Data: {data_path}")
    print(f"🤖 Model: {model_name}")
    print(f"💾 Output: {output_dir}")

    # 1. Load Model
    print("\n[1/5] Loading Model...")
    
    # Debug memory before load
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_stats.name} | VRAM: {gpu_stats.total_memory / 1024**3:.2f} GB")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        gpu_memory_utilization = 0.60, 
    )

    # 2. Add LoRA adapters
    print("\n[2/5] Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",    
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None, 
    )

    # 3. Prepare Data
    print("\n[3/5] Preparing Dataset...")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Format for Gemma-2/3 Chat
    formatted_data = []
    
    for item in raw_data:
        # Prompt + Revised
        conversation = [
            {"role": "user", "content": item.get("prompt", "")},
            {"role": "model", "content": item.get("revised_response", "")}
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        formatted_data.append({"text": text})

    dataset = Dataset.from_list(formatted_data)
    print(f"Using {len(dataset)} samples for training.")

    # 4. Train
    print("\n[4/5] Training...")
    
    # Clear cache before training loop
    torch.cuda.empty_cache()
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            warmup_steps = 5,
            max_steps = 60, 
            learning_rate = learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",
            # Explicitly enable gradient checkpointing in Trainer args too
            gradient_checkpointing = True,
        ),
    )

    trainer_stats = trainer.train()
    print(f"Training stats: {trainer_stats}")

    # 5. Save & Export
    print("\n[5/5] Saving & Exporting to GGUF...")
    
    # Save standard adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ LoRA adapters saved to {output_dir}")

    # Save GGUF (q8_0 for quality, or q4_k_m for speed)
    gguf_path = os.path.join(output_dir, "gemma_custom.gguf")
    try:
        model.save_pretrained_gguf(output_dir, tokenizer, quantization_method = "q4_k_m")
        print(f"✅ GGUF exported to {output_dir}")
    except Exception as e:
        print(f"⚠️ GGUF Export Failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_local.py <data_path> <output_dir>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Default model for now
    train_local(data_path, output_path)
