#!/usr/bin/env python3
""" Function that cleans and normalize SMS messages """


import re
import emoji

_DATASET_PLACEHOLDER_MAP = {
    "<#>": "<NUM>",
    "<decimal>": "<NUM>",
    "<time>": "<TIME>",
    "<url>": "<URL>",
    "<email>": "<EMAIL>",
}


def normalize_unicode_punct(text):
    """Replace curly quotes, dashes, ellipses, etc. with ASCII equivalents."""
    replacements = {
        r"[''‚‛]": "'",
        r"[" "„‟]": '"',
        r"[‐‑‒–—―−]": "-",
        r"…": "...",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    return text


def clean_text(
    text, replace_num=True, replace_url=True, emoji_action="replace"
):
    """ Return new_text that that is already cleaned"""
    if not isinstance(text, str):
        return ""
    # 1. lowercase + strip
    new_text = text.lower().strip()
    # 2. dataset placeholders
    for old, new in _DATASET_PLACEHOLDER_MAP.items():
        new_text = new_text.replace(old, new)
    # 3. normalize_unicode_punct()
    new_text = normalize_unicode_punct(new_text)

    # 4. URL replacement
    if replace_url:
        new_text = re.sub(r"https?://\S+|www\.\S+", "<URL>", new_text)
    # 5. number replacement (2 passes)
    if replace_num:
        new_text = re.sub(r"\+?\d[\d\s\-]{6,}\d", "<NUM>", new_text)
        new_text = re.sub(
            r"(?:£|\$|€)\d+(?:[.,]\d+)*|(?<!<)\b\d+(?:[.,]\d+)*\b",
            "<NUM>",
            new_text,
        )
    # 6. emoji handling
    if emoji_action == "replace":
        new_text = emoji.replace_emoji(new_text, replace="<EMO>")
    elif emoji_action == "remove":
        new_text = emoji.replace_emoji(new_text, replace=" ")
    # 7. collapse repeated ! / ?
    new_text = re.sub(r"!+", "!", new_text)
    new_text = re.sub(r"\?+", "?", new_text)

    # 8. collapse whitespace
    new_text = re.sub(r"\s+", " ", new_text).strip()
    return new_text
