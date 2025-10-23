import re


def extract_emojis() -> list:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
        "\U00002600-\U000026FF"  # Misc symbols
        "\U0001FA70-\U0001FAFF"  # Symbols & pictographs extended-A
        "]+",
        flags=re.UNICODE
    )

    with open("conversations.txt", "r", encoding="utf-8") as f:
        text = f.read()

    emojis = emoji_pattern.findall(text)
    emojis = sorted(set(emojis))
    emojis_extracted = "emojis_extracted.txt"

    with open(emojis_extracted, "w", encoding="utf-8") as f:
        f.write('')

    with open(emojis_extracted, "a", encoding="utf-8") as f:
        for emoji in emojis:
            f.write(f'"{emoji}",')

    print(f"✅ Extracted {len(emojis)} emojis to emojis_extracted.txt")
    return emojis
