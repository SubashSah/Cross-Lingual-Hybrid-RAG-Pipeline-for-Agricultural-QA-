import sys
from services.translation_service import translate_text

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

text = """
1. 7–10 grams
2. 7-10 grams
3. 7 to 10 grams
4. seven to ten grams
"""

translated = translate_text(text)

print("Translated Output:")
print(translated)