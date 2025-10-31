import os
import json
import shutil
import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------
# Login to Hugging Face
# ---------------------------------------------------------------------
login(token="your_hf_key")

if torch.cuda.is_available():
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Using CPU — training will be extremely slow!")

# ---------------------------------------------------------------------
# Cleanup output folder
# ---------------------------------------------------------------------
if os.path.exists("./trained_model"):
    shutil.rmtree("./trained_model")
os.makedirs("./trained_model", exist_ok=True)

# ---------------------------------------------------------------------
# Load model and tokenizer (quantized 4-bit for efficiency)
# ---------------------------------------------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------
# Fix chat template issue if needed
# ---------------------------------------------------------------------
if model_name.startswith("meta-llama") and tokenizer.chat_template is None:
    tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"""

# ---------------------------------------------------------------------
# Prepare dataset
# ---------------------------------------------------------------------
with open("conversations.json", "r") as f:
    conversations = json.load(f)

formatted_data = []
for conversation in conversations:
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    formatted_data.append(text)

dataset = Dataset.from_dict({"text": formatted_data})

def preprocess(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ---------------------------------------------------------------------
# Apply LoRA adapter (this is the key change)
# ---------------------------------------------------------------------
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                     # rank (higher = more trainable parameters)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # common for Llama models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./trained_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=50,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_32bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ---------------------------------------------------------------------
# Train LoRA adapter
# ---------------------------------------------------------------------
trainer.train()

# ---------------------------------------------------------------------
# Save model + tokenizer
# ---------------------------------------------------------------------
trainer.model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("🎉 DONE TRAINING (LoRA adapter saved in ./trained_model)")
