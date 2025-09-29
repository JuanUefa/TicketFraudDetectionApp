import re
import unicodedata
import pandas as pd
 
 
class DataCleaningUtils:
    
    def clean_names(self, name: str) -> str:
        """
        Normalize names: lowercase, strip accents, keep only letters/spaces.
        """
        if pd.isna(name) or str(name).strip() == "":
            return ""
        name = str(name).lower()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
        return re.sub(r"[^a-z\s]", "", name)
 
    
    def clean_city(self, city: str) -> str:
        """
        Normalize city names: lowercase, strip non-letters, collapse spaces.
        """
        if pd.isna(city) or str(city).strip() == "":
            return "unknown"
        city = city.lower().strip()
        city = re.sub(r"[^a-z\s]", "", city)
        return re.sub(r"\s+", " ", city)
 
    
    def clean_browser_language(self, lang: str) -> str:
        """
        Clean browser language codes like 'en-us,fr-fr' → 'en-US,fr-FR'.
        """
        if pd.isna(lang) or str(lang).strip() == "":
            return "unknown"
 
        lang_list = str(lang).lower().split(",")
        cleaned_langs = [
            re.sub(
                r"^([a-z]{2})-([a-z]{2})$",
                lambda m: f"{m.group(1)}-{m.group(2).upper()}",
                l.strip()
            )
            for l in lang_list
            if re.fullmatch(r"^[a-z]{2}-[a-z]{2}$", l.strip())
        ]
        return ",".join(cleaned_langs) if cleaned_langs else "unknown"
    

    def extract_email_components(self, email: str):
        """Extract username, domain, and TLD from an email address."""
        if pd.isna(email) or "@" not in str(email):
            return ["0", "0", "0"]
        user, domain = email.split("@", 1)
        domain_parts = domain.split(".")
        tld = domain_parts[-1] if len(domain_parts) > 1 else "0"
        return [user, domain, tld]