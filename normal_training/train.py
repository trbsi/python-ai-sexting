import json
import os
import shutil

import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, \
    Trainer, DataCollatorForLanguageModeling

# login to HuggingFace
login(token=os.getenv('HUGGING_FACE_TOKEN'))

if torch.cuda.is_available():
    print("Using GPU " + torch.cuda.get_device_name(0))

# ---------------- Remove trained model folder and recreate it ----------------
if os.path.exists('./trained_model'):
    shutil.rmtree('./trained_model')
os.makedirs('./trained_model')

# ---------- Load tokenizer and model -----------
# model_name = 'meta-llama/Llama-3.2-3B-Instruct'
# model_name = 'HuggingFaceH4/zephyr-7b-beta'
# model_name = 'google/gemma-2-2b'
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

"""
We use the "end of sentence" token for padding to make all sequences the same length
Why: So we can batch multiple conversations together during training
"""
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open('conversations.json', 'r') as file:
    conversations = json.load(file)

"""
Converts your JSON conversations into Gemma's expected chat format
"""
formatted_data = []
for conversation in conversations:
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    formatted_data.append(text)

"""
Converts your formatted texts into a Hugging Face Dataset object
Makes it compatible with their training tools
"""
dataset = Dataset.from_dict({'text': formatted_data})


def simple_preprocess(text):
    # Converts text to numbers (tokens)
    tokenized = tokenizer(text['text'], truncation=True, max_length=512, padding=False)
    # Learn to predict each next word in the sequence based on all the words that came before it
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


"""
Takes individual conversations and creates batches for training
mlm=False: We're doing causal LM (predict next word), not masked LM
pad_to_multiple_of=8: Makes GPU processing more efficient
Why batch: Training on multiple examples at once is faster
"""
tokenized_dataset = dataset.map(
    simple_preprocess,
    batched=True,
    remove_columns=dataset.column_names  # Remove original 'text' column
)

# Data collator for efficient batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling
    pad_to_multiple_of=8,
)

training_args = TrainingArguments(
    output_dir='./trained_model',
    num_train_epochs=3,  # How many times to go through your data
    per_device_train_batch_size=1,  # How many conversations per GPU
    gradient_accumulation_steps=8,  # Simulates larger batch size
    learning_rate=1e-5,  # How fast to learn (small steps)
    warmup_steps=100,  # Gradually increase learning rate at start
    weight_decay=0.01,  # Prevent overfitting
    fp16=True,  # Use less memory
    remove_unused_columns=False,  # Prevents errors

    eval_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

# --------- Save model ------------
trainer.save_model('./trained_model')
tokenizer.save_pretrained('./trained_model')

print('DONE TRAINING')
