try:
    from pyspellchecker import SpellChecker
    print("SUCCESS: pyspellchecker imported")
except Exception as e:
    print("FAILED:", e)

try:
    import Levenshtein
    print("SUCCESS: Levenshtein imported")
except Exception as e:
    print("FAILED:", e)
