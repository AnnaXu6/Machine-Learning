
import re
import string
from typing import Iterable, Optional

import pandas as pd

# Minimal English stopword list (offline fallback)
_MIN_STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
    "its","of","on","that","the","to","was","were","will","with","i","you","your",
    "we","they","them","this","these","those","not","or","but","so","if","then",
    "than","there","here","what","when","where","who","whom","why","how","can","could",
    "should","would","do","does","did","done","have","had","having","about","into","over",
    "under","again","further","once","because","until","while","both","each","few","more",
    "most","other","some","such","no","nor","only","own","same","too","very","s","t","d",
    "ll","m","o","re","ve","y","don","shouldn","now"
}

DEFAULT_DOMAIN_TERMS = [
    "#coronavirus", "#coronavirusoutbreak", "#coronaviruspandemic",
    "#covid19", "#covid_19", "#epitwitter", "#ihavecorona",
    "amp", "coronavirus", "covid19", "covid-19", "covidー19"
]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def build_stopwords(extra_terms: Optional[Iterable[str]] = None) -> set:
    sw = set(_MIN_STOPWORDS)
    sw.update(DEFAULT_DOMAIN_TERMS)
    if extra_terms:
        sw.update([t.lower() for t in extra_terms])
    return sw

def strip_urls(text: str) -> str:
    return URL_RE.sub("", text)

def strip_punct(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))

def remove_stopwords(text: str, sw: set) -> str:
    return " ".join(w for w in text.split() if w not in sw)

def clean_text_series(series: pd.Series, sw: Optional[set] = None) -> pd.Series:
    if sw is None:
        sw = build_stopwords()
    series = series.fillna("").astype(str).str.lower()
    series = series.apply(strip_urls)
    series = series.apply(strip_punct)
    series = series.apply(lambda s: remove_stopwords(s, sw))
    return series

def load_and_preprocess(csv_path: str, text_col: str = "text", lang_col: Optional[str] = "lang") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if lang_col in df.columns:
        df = df.loc[df[lang_col].astype(str).str.lower() == "en"].copy()
    df["text"] = df[text_col].fillna("").astype(str)
    sw = build_stopwords()
    df["text_clean"] = clean_text_series(df["text"], sw=sw)
    return df
