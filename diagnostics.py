# === DIAGNOSTIC: FIND WHY MODEL GENERATES "User:" IN RESPONSES ===

def diagnose_training_data():
    print("🔍 RUNNING TRAINING DATA DIAGNOSTIC\n")

    with open('conversations.txt', 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Total lines in training data: {len(lines)}")

    # 1. Check for "Bot:" lines that are followed by "User:" lines
    print("\n=== CHECKING FOR 'Bot:' → 'User:' PATTERNS ===")
    bot_to_user_transitions = 0
    for i in range(len(lines) - 1):
        if lines[i].startswith('Bot:') and lines[i + 1].startswith('User:'):
            bot_to_user_transitions += 1
            if bot_to_user_transitions <= 3:  # Show first 3 examples
                print(f"Example {bot_to_user_transitions}:")
                print(f"  {lines[i]}")
                print(f"  {lines[i + 1]}")
                print()

    print(f"Total 'Bot:' → 'User:' transitions found: {bot_to_user_transitions}")

    # 2. Check for lines that start with "Bot:" but look like user messages
    print("\n=== CHECKING FOR SUSPICIOUS 'Bot:' LINES ===")
    suspicious_bot_lines = 0
    for i, line in enumerate(lines):
        if line.startswith('Bot:') and ('User:' in line or 'user:' in line.lower()):
            suspicious_bot_lines += 1
            if suspicious_bot_lines <= 3:
                print(f"Line {i + 1}: {line}")

    print(f"Total suspicious 'Bot:' lines: {suspicious_bot_lines}")

    # 3. Check the exact pattern from your error
    print("\n=== SEARCHING FOR THE EXACT PROBLEM PATTERN ===")
    target_bot_line = "Bot: What's your name!"
    target_user_line = "User : Are u ready to get wetter than ever for me bby?"

    for i in range(len(lines) - 1):
        if (target_bot_line in lines[i] and
                target_user_line in lines[i + 1]):
            print("🚨 FOUND THE EXACT PROBLEMATIC SEQUENCE:")
            print(f"Line {i + 1}: {lines[i]}")
            print(f"Line {i + 2}: {lines[i + 1]}")
            # Show context
            if i > 0: print(f"Line {i}: {lines[i - 1]}")
            if i < len(lines) - 2: print(f"Line {i + 3}: {lines[i + 2]}")
            break
    else:
        print("Exact pattern not found, searching for similar...")
        for i, line in enumerate(lines):
            if "What's your name" in line and line.startswith('Bot:'):
                print(f"Similar pattern at line {i + 1}: {line}")
                if i < len(lines) - 1:
                    print(f"Next line: {lines[i + 1]}")

    # 4. Check data format statistics
    print("\n=== DATA FORMAT STATISTICS ===")
    user_lines = [l for l in lines if l.startswith('User:') or l.startswith('User :')]
    bot_lines = [l for l in lines if l.startswith('Bot:') or l.startswith('Bot :')]
    endoftext_lines = [l for l in lines if '<|endoftext|>' in l]

    print(f"User lines: {len(user_lines)}")
    print(f"Bot lines: {len(bot_lines)}")
    print(f"Endoftext markers: {len(endoftext_lines)}")
    print(f"Other lines: {len(lines) - len(user_lines) - len(bot_lines) - len(endoftext_lines)}")

    # 5. Check conversation flow
    print("\n=== CONVERSATION FLOW ANALYSIS ===")
    current_speaker = None
    broken_flows = 0

    for i, line in enumerate(lines):
        if '<|endoftext|>' in line:
            current_speaker = None
            continue

        if line.startswith('User:') or line.startswith('User :'):
            if current_speaker == 'User':
                broken_flows += 1
                if broken_flows <= 2:
                    print(f"Broken flow at line {i + 1}: User spoke twice in a row")
            current_speaker = 'User'
        elif line.startswith('Bot:') or line.startswith('Bot :'):
            if current_speaker == 'Bot':
                broken_flows += 1
                if broken_flows <= 2:
                    print(f"Broken flow at line {i + 1}: Bot spoke twice in a row")
            current_speaker = 'Bot'

    print(f"Total broken conversation flows: {broken_flows}")

    return lines


# Check if this is the issue
def check_for_contaminated_responses():
    with open('conversations.txt', 'r') as f:
        lines = f.readlines()

    print("Searching for bot responses that contain 'User:'...")
    contaminated_count = 0

    for i, line in enumerate(lines):
        if line.startswith('Bot:') and 'User:' in line:
            contaminated_count += 1
            if contaminated_count <= 5:  # Show first 5 examples
                print(f"Line {i + 1}: {line.strip()}")

    print(f"\nTotal contaminated bot responses: {contaminated_count}")
    return contaminated_count


contaminated = check_for_contaminated_responses()

if contaminated > 0:
    print("🚨 CONFIRMED: Your training data has bot responses that contain 'User:' text!")
    print("This is why your model generates 'User:' in its replies.")
else:
    print("The issue might be different - let me check another possibility...")


# RUN THE DIAGNOSTIC
training_data_lines = diagnose_training_data()

print("\n" + "=" * 50)
print("📋 SUMMARY: The diagnostic will show you exactly where")
print("your training data has 'Bot:' lines that are immediately")
print("followed by 'User:' lines, which teaches the model to")
print("generate 'User:' in its responses!")
print("=" * 50)