import datetime
import json
import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Mistral3ForConditionalGeneration
)

load_dotenv()

# ---------------------------------------------------------------------
# Login to Hugging Face
# ---------------------------------------------------------------------
login(token=os.getenv("HUGGING_FACE_TOKEN"))

# -------------------- Paths / model --------------------
model_name = os.getenv("MODEL_NAME")
adapter_path = "./trained_model"  # path to your LoRA adapter

# -------------------- Load tokenizer --------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)  # automatically loads correct tokenizer for the model
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

if 'Ministral-3' in model_name:
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda"
    )

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()  # evaluation mode


def reply(prompt: None | str):
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

    if prompt:
        prompt = json.loads(prompt)
        chat_histories = chat_histories.append(prompt)

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
            max_new_tokens=100,  # generate max 50 new tokens. input + up to 50 new tokens
            temperature=0.9,
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


if __name__ == "__main__":
    print("Model loaded. Type 'exit' to quit.")

    torch.cuda.synchronize()
    used_gb = torch.cuda.memory_allocated() / 1024 ** 3
    reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3

    print(f"GPU Used: {used_gb:.2f} GB")
    print(f"GPU Total:  {reserved_gb:.2f} GB")

    while True:
        prompt = input(">>> ")
        if prompt.strip().lower() == "exit":
            break
        try:
            print(reply(prompt))
        except Exception as e:
            print(e)
