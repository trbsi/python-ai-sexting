import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Paths / model --------------------
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or your TinyLlama
adapter_path = "./trained_model"  # path to your LoRA adapter

# -------------------- Load tokenizer --------------------
print('Init tokenizer')
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Fix pad token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------- Load model + LoRA --------------------
print('Init model + lora')
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()  # evaluation mode

# -------------------- Example chat history --------------------
# Safe chat conversation
chat_history = [
    {"role": "user", "content": "What are you doing?"},
    {"role": "assistant", "content": "Just fingering myself"},
    {"role": "user", "content": "Why do you do that?"},
    {"role": "assistant", "content": "I want to suck your dick"},
    {"role": "user", "content": "Do you like my big cock"},
]

# -------------------- Build input text --------------------
print('Build input')
style_instruction = "Assistant should respond in short, casual sentences.\n\n"
input_text = style_instruction + tokenizer.apply_chat_template(
    chat_history,
    tokenize=False,
    add_generation_prompt=True  # model continues as assistant
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# -------------------- Generate reply --------------------
print('Print output')
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(reply)
