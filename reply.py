import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def reply(chat_history: str) -> str:
    """
    :param chat_history: in format like below in example, messages are one blow other
    """
    inputs = tokenizer.encode(chat_history, return_tensors='pt')
    if torch.cuda.is_available():
        inputs.to('cuda')
        model.to('cuda')

    bad_words = ["User:", "Bot:", "<|endoftext|>", "User :", "Bot :"]
    bad_words_ids = [tokenizer.encode(word)[0] for word in bad_words if tokenizer.encode(word)]

    reply_ids = model.generate(
        inputs,
        max_length=128,
        pad_token_id=tokenizer.eos_token_id,
        top_p=0.9,  # Keeps top 90% probability mass — more natural replies
        temperature=0.8,  # smooths token probabilities
        repetition_penalty=1.3,  # Penalizes repeating tokens (like “...”)
        no_repeat_ngram_size=3,  # prevents repeating same 3-gram phrases
        bad_words_ids=[bad_words_ids] if bad_words_ids else None,
    )
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)


model_names = [
    # 'microsoft/DialoGPT-small',
    # 'microsoft/DialoGPT-medium',
    # 'microsoft/DialoGPT-large',
    './trained_model',
]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    chat_history = [
        'User: Hello sweetheart. Please reply.',
        'Bot: So hi, my fucker, how are you.',
        'User: Bad.',
        'Bot: Why are you bad honey?',
        'User: I haven\'t fucked in a long time.',
        'Bot: Oh, fuck. So how long haven\'t you fucked?',
        'User: A dozen days and you\'ve been fucking lately or are you dry?',
        'Bot:'
    ]
    result = reply('\n'.join(chat_history))
    print(model_name)
    print(result)
    print('')
