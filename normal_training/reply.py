import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def reply(chat_history: list, model, tokenizer, use_cuda: bool) -> str:
    """
    :param chat_history: in format  [{"role": "user", "content": "some message"}]
    """

    print('Apply chat template')
    prompt = tokenizer.apply_chat_template(
        chat_history,
        tokenize=False,  # Returns a text string instead of token IDs
        add_generation_prompt=True,  # Adds a special token that tells the model "start generating now"
    )

    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"

    # Tokenize - using the CORRECT method
    print('Tokenizer')
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    print('Generate output')
    with torch.no_grad():  # Disables gradient calculation (faster, uses less memory)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # ↑ Use greedy search (faster)
            temperature=1.0,  # ↓ Lower = faster but less creative
            num_beams=1,  # ↑ No beam search (much faster)
            early_stopping=True,  # ↑ Stop when confident
            pad_token_id=tokenizer.eos_token_id,
        )

    print('Decode response')
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# device = 'cpu'
model_path = './trained_model'
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name='./trained_model',
#     max_seq_length=2048,
#     dtype=None,
#     load_in_4bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",  # explicitly use CPU
    offload_folder="offload",
    low_cpu_mem_usage=True,
)
chat_history = [
    {"role": "user", "content": "Hello sweetheart. Please reply."},
    {"role": "assistant", "content": "So hi, my fucker, how are you?"},
    {"role": "user", "content": "Bad"},
    {"role": "assistant", "content": "Why are you bad honey?"},
    {"role": "user", "content": "I haven't fucked in a long time"},
    {"role": "assistant", "content": "Oh, fuck. So how long haven't you fucked?"},
    {"role": "user", "content": "A dozen days and you've been fucking lately or are you dry?"},
]

start_time = time.time()
result = reply(chat_history, model, tokenizer, False)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

# print(type(model))
# print(type(tokenizer))
print(result)
print('')
