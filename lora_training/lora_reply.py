import datetime
import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

# -------------------- Paths / model --------------------
base_model = os.getenv("MODEL_NAME")
adapter_path = "./trained_model"  # path to your LoRA adapter

# -------------------- Load tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Fix pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- Load model + LoRA --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="cuda"
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()  # evaluation mode

# -------------------- Example chat history --------------------
# Safe chat conversation
chat_histories = [
    [
        {"role": "user", "content": "What are you doing?"},
        {"role": "assistant", "content": "Just fingering myself"},
        {"role": "user", "content": "Why do you do that?"},
        {"role": "assistant", "content": "I want to suck your dick"},
        {"role": "user", "content": "Do you like my big cock"},
    ],
    [
        {"role": "user", "content": "What are you doing?"},
    ],
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "What's up?"},
    ],
    [
        {"role": "user", "content": "Morning"},
        {"role": "assistant", "content": "I'm horny"},
        {"role": "user", "content": "I'm horny to uffff"},
        {"role": "assistant", "content": "Suck my big fucking asshole"},
        {"role": "user", "content": "Suck my big fucking dick"},
    ]
]

for chat_history in chat_histories:
    t1 = datetime.datetime.now().timestamp()
    # -------------------- Build input text --------------------
    chat_history.insert(
        0,
        {
            "role": "system",
            "content": "Assistant should respond in short, casual sentences. Talk nasty and dirty. Ask user some questions. Engage user in conversation."
        },
    )
    input_text = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,
        add_generation_prompt=True  # model continues as assistant
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # -------------------- Generate reply --------------------
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  # generate max 50 new tokens. input + up to 50 new tokens
        temperature=0.7,
        # controls randomness, lower value means safe/precise generation, higher value gives more randomness and creativity
        top_p=0.9,  #
        do_sample=True,  # I don't understand this
        pad_token_id=tokenizer.eos_token_id,
        # some models don't have pad token, but it might be required during generation
        eos_token_id=tokenizer.eos_token_id  # if model generates OES generation stops early
    )
    input_length = inputs['input_ids'].shape[1]  # get size of input text
    generated_tokens = outputs[0][input_length:]  # basically remove input tokens and get only newly generated
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    t2 = datetime.datetime.now().timestamp()
    time = round(t2 - t1)
    print(f'{time} seconds')
    print("Reply:", reply)
