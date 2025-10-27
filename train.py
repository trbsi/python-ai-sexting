import json
import os
import shutil

import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, TrainingArguments, \
    Trainer
from transformers.models.auto.modeling_auto import _BaseModelWithGenerate


def format_training_data(conversations: list) -> list:
    formatted_data = []
    for conversation in conversations:
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        formatted_data.append({'text': text})

    return formatted_data


# login to HuggingFace
login(token=env)

if torch.cuda.is_available():
    print("Using GPU " + torch.cuda.get_device_name(0))

# ---------------- Remove trained model folder and recreate it ----------------
if os.path.exists('./trained_model'):
    shutil.rmtree('./trained_model')
os.makedirs('./trained_model')

# ---------- Load tokenizer and model -----------
model_name = 'meta-llama/Llama-4-Scout-17B-16E-Instruct'
# model_name = 'google/gemma-3n-E4B-it'
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: _BaseModelWithGenerate = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

print(type(tokenizer))
print(type(model))

with open('conversations.json', 'r') as f:
    conversations = json.load(f.read())

formatted_data = format_training_data(conversations)
dataset = Dataset.from_list(formatted_data)

print(formatted_data)

training_args = TrainingArguments(
    output_dir='./trained_model',
    learning_rate=1e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# --------- Save model ------------
trainer.save_model('./trained_model')
tokenizer.save_pretrained('./trained_model')

print('DONE TRAINING')
