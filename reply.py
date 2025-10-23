import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('./trained_model')
model = AutoModelForCausalLM.from_pretrained('./trained_model')


def reply(chat_history: str) -> str:
    """
    :param chat_history: in format like below in example, messages are one blow other
    """
    inputs = tokenizer.encode(chat_history, return_tensors='pt')
    if torch.cuda.is_available():
        inputs.to('cuda')
        model.to('cuda')

    reply_ids = model.generate(
        inputs,
        max_length=120,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,  # Keeps top 90% probability mass — more natural replies
        temperature=0.8,  # smooths token probabilities
        repetition_penalty=1.2,  # Penalizes repeating tokens (like “...”)
        no_repeat_ngram_size=3,  # prevents repeating same 3-gram phrases
    )
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)


chat_history = ['User: How are you today?', 'Bot:']
result = reply('\n'.join(chat_history))
print(result)
