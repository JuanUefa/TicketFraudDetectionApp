import re
import zlib
import numpy as np
from difflib import SequenceMatcher
from nltk.util import ngrams
from scipy.stats import entropy
 
 
class DataTransformationUtils:

    def __init__(self):
        pass
    
    def count_digits(self, text: str) -> int:
        if not text:
            return 0
        return sum(c.isdigit() for c in str(text))
 
    
    def count_special_chars(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"[\W_]", str(text)))
 
    
    def count_unique_chars(self, text: str) -> int:
        return len(set(str(text))) if text else 0
 
    
    def digit_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        return sum(c.isdigit() for c in text) / len(text)
 
    
    def special_char_ratio(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        return sum(not c.isalnum() for c in text) / len(text)
 
    
    def kolmogorov_complexity(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        return len(zlib.compress(text.encode())) / len(text)
 
    
    def text_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        text = str(text)
        char_counts = np.array([text.count(c) for c in set(text)])
        return entropy(char_counts) / len(text) if len(text) > 0 else 0.0
 
    
    def count_numeric_sequences(self, username: str) -> int:
        if username is None or not isinstance(username, str):
            return 0
        start_numbers = re.match(r"^(\d+)", username)
        start_count = 1 if start_numbers else 0
        username_no_end = re.sub(r"\d+$", "", username)
        middle_numbers = re.findall(r"\d+", username_no_end)
        return start_count + len(middle_numbers)
 
    # --- Helpers for name matching ---
    
    def _generate_ngrams(self, text: str, n: int = 3):
        text = re.sub(r"[^a-zA-Z]", "", str(text).lower())
        return {"".join(g) for g in ngrams(text, n)}
 
    
    def _fuzzy_match(self, name: str, username: str, threshold: float = 0.8) -> bool:
        if not name or not username:
            return False
        for i in range(len(username) - len(name) + 1):
            substring = username[i : i + len(name)]
            similarity = SequenceMatcher(None, name, substring).ratio()
            if similarity >= threshold:
                return True
        return False
 
    
    def match_name_in_username(self, first_name: str, last_name: str, email_username: str) -> dict:
        """
        Returns a dict with exact, ngram, and fuzzy match flags.
        """
        try:
            username = str(email_username).lower()
            first = str(first_name).lower()
            last = str(last_name).lower()
 
            first_exact = first in username
            last_exact = last in username
 
            username_ngrams = self._generate_ngrams(username)
            first_ngrams = self._generate_ngrams(first)
            last_ngrams = self._generate_ngrams(last)
 
            first_ngram = not username_ngrams.isdisjoint(first_ngrams)
            last_ngram = not username_ngrams.isdisjoint(last_ngrams)
 
            first_fuzzy = self._fuzzy_match(first, username)
            last_fuzzy = self._fuzzy_match(last, username)
 
            return {
                "first_name_exact_match": first_exact,
                "last_name_exact_match": last_exact,
                "first_name_ngram_match": first_ngram,
                "last_name_ngram_match": last_ngram,
                "first_name_fuzzy_match": first_fuzzy,
                "last_name_fuzzy_match": last_fuzzy,
            }
        except Exception:
            return {
                "first_name_exact_match": False,
                "last_name_exact_match": False,
                "first_name_ngram_match": False,
                "last_name_ngram_match": False,
                "first_name_fuzzy_match": False,
                "last_name_fuzzy_match": False,
            }
 
    
    def inverse_semantic_score(self, first_name: str, last_name: str, email_username: str) -> int:
        """
        Returns a score = number of mismatched match flags (0 = perfect, 6 = worst).
        """
        try:
            username = str(email_username).lower()
            first = str(first_name).lower()
            last = str(last_name).lower()
 
            exact1 = first in username
            exact2 = last in username
 
            grams_user = self._generate_ngrams(username)
            grams_first = self._generate_ngrams(first)
            grams_last = self._generate_ngrams(last)
 
            ngram1 = not grams_user.isdisjoint(grams_first)
            ngram2 = not grams_user.isdisjoint(grams_last)
 
            fuzzy1 = self._fuzzy_match(first, username)
            fuzzy2 = self._fuzzy_match(last, username)
 
            match_flags = [exact1, exact2, ngram1, ngram2, fuzzy1, fuzzy2]
            return len(match_flags) - sum(match_flags)
        
        except Exception:
            return 6