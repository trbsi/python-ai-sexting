import json
import os
import shutil

import torch
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend, BitsAndBytesConfig
)

load_dotenv()

# ---------------------------------------------------------------------
# Login to Hugging Face
# ---------------------------------------------------------------------
login(token=os.getenv("HUGGING_FACE_TOKEN"))

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
model_name = os.getenv("MODEL_NAME")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

if 'Ministral-3' in model_name:
    tokenizer = MistralCommonBackend.from_pretrained(model_name)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # automatically loads correct tokenizer for the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",  # determine automatically which GPU/CPU the model is loaded on
        quantization_config = bnb_config,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------
# Prepare dataset
# ---------------------------------------------------------------------
"""
conversations.json
[
  [
    {
      "role": "user",
      "content": "Hello. Please reply."
    },
    {
      "role": "assistant",
      "content": "So hi, how are you."
    }
  ]
]
"""
with open("conversations.json", "r") as f:
    conversations = json.load(f)

"""
Formatted data will be array of strings for each conversation:
"<|system|> You are a helpful assistant.\n<|user|> Hello?\n<|assistant|>",
"<|system|> You are a helpful assistant.\n<|user|> Hello, who won the World Cup in 2022?\n<|assistant|>"
"""
formatted_data = []
for conversation in conversations:
    try:
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,  # do not tokenize just yet, return normal string
            add_generation_prompt=True  # add "assistant" prompt at the end: <|assistant|>
        )
        formatted_data.append(text)
    except Exception as e:
        print(e)
        print(conversation)

"""
creates HuggingFace Dataset object from list.
Output for dataset is:
Dataset({
    features: ['text'],
    num_rows: 1000
})
dataset[0]["text"] equals to "<|system|> You are a helpful assistant.\n<|user|> Hello?\n<|assistant|>"
"""
dataset = Dataset.from_dict({"text": formatted_data})


def preprocess(example):
    """
    convert conversation to tokens
    tokenized = {
        "input_ids": [15496, 11, 995, 50256],
        "attention_mask": [1, 1, 1, 1],
        "labels": [15496, 11, 995, 50256],
    }
    """
    tokenized = tokenizer(
        example["text"],
        truncation=True,  # cut off sequences longer than max_length
        max_length=512,  # maximum number of tokens in input_ids
        padding=False,  # do not pad sequences, it will be done in batch collator
    )
    # in causal model target lables are the same as input IDs
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# tokenized HuggingFace Dataset with columns: ["input_ids", "attention_mask", "labels"]
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,  # process multiple examples at once
    remove_columns=["text"]  # remove original "text" column after tokenization
)

# ---------------------------------------------------------------------
# Apply LoRA adapter (this is the key change)
# ---------------------------------------------------------------------
model.gradient_checkpointing_enable()  # don't understand what it does honestly
model = prepare_model_for_kbit_training(model)  # just prepares quantized model for safe training

lora_config = LoraConfig(
    r=16,  # rank (higher = more trainable parameters). How complex model update can be.
    lora_alpha=32,  # how strong the update is
    target_modules=["q_proj", "v_proj"],
    # which layers to modify, query and value projections. they have the most influence on attention behviour.
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Prints something like:
# trainable params: 4,194,304 || all params: 6,738,415,616 || trainable: 0.062%
model.print_trainable_parameters()

# ---------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./trained_model",
    num_train_epochs=3,  # number of full passes over dataset (2-5 epoch)
    per_device_train_batch_size=1,  # how many training samples are processed at once on single GPU
    gradient_accumulation_steps=8,
    # process 8 samples before updating weights and resetting gradients (the “direction and strength” in which each weight should change to make the model better.)
    learning_rate=1e-4,  # how fast model learns
    warmup_steps=50,  # do not learn all at once, learn gradually
    fp16=True,  # use 16-bit floating number during training
    logging_steps=10,  # log training loss every 10 steps
    save_strategy="epoch",  # save a checkpoint after each epoch
    report_to="none",
    optim="paged_adamw_32bit",  # This optimizer is from bitsandbytes
)

# The data collator takes the tokenized dataset samples and turns them into PyTorch tensors with the correct shape so the model can understand them.
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # this means we are doing causal language modeling (MLM=masked e.g. BERT, CLM=causal e.g. GPT)
    pad_to_multiple_of=8,
    # instead of padding to the longest sequence, pad to next multiple of 8 tokens. mostly because of GPU effciency
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
