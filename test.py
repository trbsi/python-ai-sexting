import json


def reply(p):
    print(json.loads(p))


if __name__ == "__main__":
    print("Model loaded. Type 'exit' to quit.")
    while True:
        prompt = input(">>> ")
        if prompt.strip().lower() == "exit":
            break
        print(reply(prompt))
