
def create_validation_data():
    with open('conversations.txt', 'r') as f:
        all_conversations = f.readlines()

    # Split 90% train, 10% validation
    train_data, validation_data = train_test_split(all_conversations, test_size=0.1, random_state=42)

    with open('validation_data.txt', 'w') as f:
        f.writelines(validation_data)


    general_questions = [
        "User: What's 2+2?\nBot: 2+2 equals 4.\n<|endoftext|>\n",
        "User: What is the capital of France?\nBot: The capital of France is Paris.\n<|endoftext|>\n",
        "User: Hello, how are you?\nBot: I'm doing well, thank you for asking!\n<|endoftext|>\n",
        "User: What color is the sky?\nBot: The sky appears blue during the day.\n<|endoftext|>\n"
    ]

    with open('validation_data.txt', 'a') as f:  # 'a' means append
        for q in general_questions:
            f.write(q)