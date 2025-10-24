import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, GPT2LMHeadModel, TrainingArguments, \
    Trainer
import os
import shutil

from extract_emojis import extract_emojis
from validation import create_validation_data

if torch.cuda.is_available():
    print("Using GPU " + torch.cuda.get_device_name(0))

# ---------------- Remove trained model folder and recreate it ----------------
if os.path.exists('./trained_model'):
    shutil.rmtree('./trained_model')
os.makedirs('./trained_model')

# ---------------- Prepare validation data -----------------
create_validation_data()

# ---------- Load tokenizer and model -----------
model_name = 'microsoft/DialoGPT-large'
tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

emojis = extract_emojis()
print(emojis)
tokenizer.add_tokens(emojis)
model.resize_token_embeddings(len(tokenizer))

# ---------- Prepare dataset --------------
dataset = load_dataset(
    'text',
    data_files={
        'chat_train': 'conversations.txt',
        'validation': 'validation_data.txt'
    }
)


# Apply tokenizer for each text entry
def preprocess(training_text):
    encodings = tokenizer(
        training_text['text'],  # raw text
        truncation=True,  # cut off if it's longer than max_length
        padding="max_length",  # Pad shorter text with zeros
        max_length=128  # Fixed token length of tokens (words) for uniform batches
    )
    encodings['labels'] = encodings['input_ids'].copy()
    return encodings


tokenized = dataset.map(preprocess, batched=True)

# ------------ Start training --------------
training_args = TrainingArguments(
    output_dir='./trained_model',  # Folder where checkpoints and the final model are saved.
    num_train_epochs=2,  # how many times the model will “read through” your dataset during training.
    per_device_train_batch_size=2,  # How many examples are processed on each GPU at a time
    save_steps=500,  # Save a checkpoint every N steps. Useful to avoid losing progress.
    save_total_limit=2,  # Keep only the latest N checkpoints to save disk space.
    logging_steps=500,  # How often to print training loss / metrics.
    report_to='none',  # Where to send logs.
    learning_rate=3e-5,

    # VALIDATION SETTINGS
    eval_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",  # Save model after each epoch
    load_best_model_at_end=True,  # Keep the BEST model, not the last
    metric_for_best_model="eval_loss",  # Use validation loss to choose best

    warmup_steps=100,  # Gradual start
    weight_decay=0.01,  # Regularization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['chat_train'],
    eval_dataset=tokenized['validation']
)
trainer.train()

# --------- Save model ------------
trainer.save_model('./trained_model')
tokenizer.save_pretrained('./trained_model')

print('DONE TRAINING')