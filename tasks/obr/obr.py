import re

from pybraille import convertText

# Function to filter text for Braille conversion
def filter_supported_characters(text: str) -> str:
    """Filters text for Braille conversion."""
    filtered_text = ""
    for char in text:
        try:
            # Test if character can be converted
            braille_char = convertText(char)
            if braille_char is not None:
                filtered_text += char
        except Exception:
            pass
    return filtered_text

def clean_text_for_braille(text: str) -> str:
    """Cleans text for Braille conversion."""
    return re.sub(r"[^a-zA-Z0-9 .,!?\'\"\-\n]", "", text)

def convert_to_braille(text: str) -> str:
    """Converts text to Braille."""
    braille_text: str = ""
    for char in text:
        braille_char = convertText(char)
        if braille_char is not None:
            braille_text += braille_char
        else:
            braille_text += "⍰"  # Use placeholder for unsupported characters
    return braille_text
