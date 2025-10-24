# Test your current model to see the extent of damage
import torch
from transformers import AutoModel, AutoModelForCausalLM

from reply import tokenizer


def test_knowledge():
    model = AutoModelForCausalLM.from_pretrained('./trained_model')
    test_questions = [
        "What's 2+2?",
        "What is the capital of France?",
        "Hello, how are you?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?"
    ]

    for question in test_questions:
        inputs = tokenizer.encode(question, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Q: {question}")
        print(f"A: {response}")
        print("---")


test_knowledge()