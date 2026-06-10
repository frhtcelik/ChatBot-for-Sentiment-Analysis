"""Hybrid sentiment analysis utilities.

The project keeps TextBlob as a simple baseline, then improves it with a
small Turkish/English lexicon so the demo works better for local use cases.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from pathlib import Path
from textblob import TextBlob


TOKEN_PATTERN = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ']+")

POSITIVE_TERMS: Dict[str, float] = {
    "iyi": 0.65,
    "iyiydi": 0.65,
    "iyidir": 0.65,
    "iyiydim": 0.65,
    "guzel": 0.72,
    "güzel": 0.72,
    "güzeldi": 0.72,
    "harika": 0.9,
    "harikaydı": 0.9,
    "harikaydim": 0.9,
    "mukemmel": 0.95,
    "mükemmel": 0.95,
    "mükemmeldi": 0.95,
    "basarili": 0.75,
    "başarılı": 0.75,
    "başarı": 0.72,
    "basari": 0.72,
    "mutlu": 0.75,
    "mutluyum": 0.78,
    "seviyorum": 0.82,
    "seviyor": 0.8,
    "sevdim": 0.72,
    "begendim": 0.74,
    "beğendim": 0.74,
    "begeniyorum": 0.8,
    "beğeniyorum": 0.8,
    "bayıldım": 0.9,
    "bayildim": 0.9,
    "hoşlanıyorum": 0.8,
    "hoslaniyorum": 0.8,
    "keyifli": 0.68,
    "keyif": 0.65,
    "memnun": 0.78,
    "memnunum": 0.8,
    "olumlu": 0.72,
    "süper": 0.85,
    "super": 0.85,
    "muhteşem": 0.92,
    "muhtesem": 0.92,
    "şahane": 0.88,
    "sahane": 0.88,
    "enfes": 0.85,
    "çok": 0.15,
    "cok": 0.15,
    "awesome": 0.9,
    "excellent": 0.92,
    "great": 0.82,
    "good": 0.62,
    "love": 0.85,
    "happy": 0.75,
    "amazing": 0.9,
    "perfect": 0.95,
    "wonderful": 0.88,
}

NEGATIVE_TERMS: Dict[str, float] = {
    "kotu": -0.72,
    "kötü": -0.72,
    "berbat": -0.92,
    "rezalet": -0.95,
    "korkunç": -0.88,
    "korkunc": -0.88,
    "dehşet": -0.85,
    "dehset": -0.85,
    "felaket": -0.9,
    "feci": -0.82,
    "mutsuz": -0.78,
    "uzgun": -0.72,
    "üzgün": -0.72,
    "üzücü": -0.75,
    "uzucu": -0.75,
    "sevmedim": -0.78,
    "sevmiyorum": -0.8,
    "sevemedim": -0.75,
    "beğenmedim": -0.75,
    "begenmedim": -0.75,
    "beğenmiyorum": -0.8,
    "hoşlanmıyorum": -0.8,
    "hoslanmiyorum": -0.8,
    "nefret": -0.9,
    "yorgun": -0.5,
    "sikildim": -0.68,
    "sıkıldım": -0.68,
    "basarisiz": -0.74,
    "başarısız": -0.74,
    "olumsuz": -0.7,
    "sinirli": -0.72,
    "sinir": -0.68,
    "kizgin": -0.74,
    "kızgın": -0.74,
    "üzgünüm": -0.76,
    "pişman": -0.7,
    "pisман": -0.7,
    "bad": -0.68,
    "terrible": -0.92,
    "awful": -0.9,
    "hate": -0.9,
    "sad": -0.74,
    "angry": -0.74,
    "poor": -0.62,
    "worst": -0.95,
    "horrible": -0.88,
}

NEGATORS = {
    "degil", "değil", 
    "not", "never", "no",
    "yok", "hiç", "hic",
    # Olumsuzluk ekleri (suffix patterns)
    "me", "ma",  # -me/-ma ekleri
    "mez", "maz",  # -mez/-maz
    "mıyor", "miyor", "muyor", "müyor",  # -mıyor/-miyor/-muyor/-müyor
    "mıyorum", "miyorum", "muyorum", "müyorum",  # -mıyorum vb.
    "madım", "medim", "madik", "medik",  # geçmiş zaman olumsuz
    "mam", "mem", "mıyım", "miyim",  # şimdiki zaman olumsuz
}


@dataclass(frozen=True)
class SentimentResult:
    text: str
    label: str
    score: float
    subjectivity: float
    confidence: float
    textblob_score: float
    lexicon_score: float
    matched_terms: List[str]

    @property
    def label_tr(self) -> str:
        labels = {
            "positive": "Pozitif",
            "neutral": "Notr",
            "negative": "Negatif",
        }
        return labels[self.label]


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _term_score(tokens: Iterable[str]) -> tuple[float, List[str]]:
    token_list = list(tokens)
    scores: List[float] = []
    matched_terms: List[str] = []

    # Sort keys by length descending to match the longest stem first
    pos_keys = sorted(POSITIVE_TERMS.keys(), key=len, reverse=True)
    neg_keys = sorted(NEGATIVE_TERMS.keys(), key=len, reverse=True)
    
    # Check if sentence contains negation suffixes
    negative_suffixes = ["miyorum", "mıyorum", "muyorum", "müyorum",
                        "miyor", "mıyor", "muyor", "müyor",
                        "mem", "mam", "mez", "maz",
                        "madım", "medim", "madik", "medik",
                        "mıyım", "miyim", "muyum", "müyüm"]
    
    has_sentence_negation = any(
        any(token.endswith(suffix) and len(token) > len(suffix) 
            for suffix in negative_suffixes)
        for token in token_list
    )

    for index, token in enumerate(token_list):
        value = None
        matched_key = None

        # 1. Exact match check
        if token in POSITIVE_TERMS:
            value = POSITIVE_TERMS[token]
            matched_key = token
        elif token in NEGATIVE_TERMS:
            value = NEGATIVE_TERMS[token]
            matched_key = token
        else:
            # 2. Prefix match check (since Turkish is suffix-heavy, e.g. "güzeldi" starts with "güzel")
            for pk in pos_keys:
                if token.startswith(pk):
                    value = POSITIVE_TERMS[pk]
                    matched_key = pk
                    break
            if not value:
                for nk in neg_keys:
                    if token.startswith(nk):
                        value = NEGATIVE_TERMS[nk]
                        matched_key = nk
                        break

        if matched_key is None or value is None:
            continue

        # Check for negation in previous tokens (e.g., "değil iyi")
        previous_tokens = set(token_list[max(0, index - 2) : index])
        has_local_negation = bool(previous_tokens & NEGATORS)
        
        # Also check if previous tokens start with "değil" (değilim, değildi, etc.)
        if not has_local_negation:
            has_local_negation = any(
                t.startswith("değil") or t.startswith("degil") 
                for t in previous_tokens
            )
        
        # Check next tokens for negation (e.g., "mutlu değilim")
        if not has_local_negation:
            next_tokens = set(token_list[index + 1 : min(len(token_list), index + 3)])
            has_local_negation = bool(next_tokens & NEGATORS)
            if not has_local_negation:
                has_local_negation = any(
                    t.startswith("değil") or t.startswith("degil")
                    for t in next_tokens
                )
        
        # Check for Turkish negative suffixes in the current token
        has_suffix_negation = any(
            token.endswith(suffix) and len(token) > len(suffix)
            for suffix in negative_suffixes
        )
        
        # If sentence has negation or local negation, flip the sentiment
        if has_sentence_negation or has_local_negation or has_suffix_negation:
            value *= -0.9

        scores.append(value)
        matched_terms.append(f"{token} ({matched_key})")

    if not scores:
        return 0.0, []

    average = sum(scores) / len(scores)
    density_boost = min(0.18, math.log1p(len(scores)) / 10)
    boosted = average + density_boost if average > 0 else average - density_boost
    return max(-1.0, min(1.0, boosted)), matched_terms


def _label_from_score(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


import pickle

MODEL_PATH = Path("data/sentiment_model.pkl")
VECTORIZER_PATH = Path("data/tfidf_vectorizer.pkl")

_model = None
_vectorizer = None


def _load_ml_model() -> None:
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    _vectorizer = pickle.load(f)
            except Exception:
                pass


def reload_ml_model() -> None:
    """Force reload the ML model from disk."""
    global _model, _vectorizer
    _model = None
    _vectorizer = None
    _load_ml_model()


def analyze_sentiment(text: str, model_type: str = "hybrid") -> SentimentResult:
    """Analyze a single text and return a normalized sentiment result."""
    cleaned = text.strip()
    if not cleaned:
        return SentimentResult(
            text=text,
            label="neutral",
            score=0.0,
            subjectivity=0.0,
            confidence=0.0,
            textblob_score=0.0,
            lexicon_score=0.0,
            matched_terms=[],
        )

    # If ML model is selected and available, use it
    if model_type == "ml":
        _load_ml_model()
        if _model is not None and _vectorizer is not None:
            from app.train_model import clean_text
            cleaned_ml = clean_text(cleaned)
            if cleaned_ml:
                vec = _vectorizer.transform([cleaned_ml])
                prediction = _model.predict(vec)[0]
                probs = _model.predict_proba(vec)[0]
                classes = _model.classes_
                
                class_idx = list(classes).index(prediction)
                confidence = probs[class_idx]
                
                pos_idx = list(classes).index("positive") if "positive" in classes else -1
                neg_idx = list(classes).index("negative") if "negative" in classes else -1
                pos_prob = probs[pos_idx] if pos_idx != -1 else 0.0
                neg_prob = probs[neg_idx] if neg_idx != -1 else 0.0
                score = pos_prob - neg_prob
                
                # Identify matched vocabulary terms
                feature_names = _vectorizer.get_feature_names_out()
                matched_terms = [w for w in cleaned_ml.split() if w in feature_names]
                
                return SentimentResult(
                    text=cleaned,
                    label=str(prediction),
                    score=round(float(score), 4),
                    subjectivity=0.5,
                    confidence=round(float(confidence), 4),
                    textblob_score=0.0,
                    lexicon_score=round(float(score), 4),
                    matched_terms=matched_terms,
                )

    blob = TextBlob(cleaned)
    textblob_score = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    lexicon_score, matched_terms = _term_score(tokenize(cleaned))

    if matched_terms and abs(textblob_score) > 0.05:
        score = (lexicon_score * 0.6) + (textblob_score * 0.4)
    elif matched_terms:
        score = lexicon_score
    else:
        score = textblob_score

    score = max(-1.0, min(1.0, score))
    label = _label_from_score(score)
    confidence = min(0.98, 0.45 + (abs(score) * 0.38) + (min(len(matched_terms), 5) * 0.03))

    if matched_terms and subjectivity < 0.25:
        subjectivity = 0.45

    return SentimentResult(
        text=cleaned,
        label=label,
        score=round(score, 4),
        subjectivity=round(subjectivity, 4),
        confidence=round(confidence, 4),
        textblob_score=round(textblob_score, 4),
        lexicon_score=round(lexicon_score, 4),
        matched_terms=matched_terms,
    )


def analyze_many(texts: Iterable[str], model_type: str = "hybrid") -> List[SentimentResult]:
    return [analyze_sentiment(text, model_type) for text in texts]


def get_word_sentiments(text: str, model_type: str = "hybrid") -> List[tuple[str, float]]:
    """Get sentiment contribution scores for individual words/tokens in the text."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if model_type == "ml":
        _load_ml_model()
        if _model is not None and _vectorizer is not None:
            from app.train_model import clean_text
            cleaned_ml = clean_text(cleaned)
            words = cleaned_ml.split()
            classes = list(_model.classes_)
            pos_idx = classes.index("positive") if "positive" in classes else -1
            neg_idx = classes.index("negative") if "negative" in classes else -1
            
            coef = _model.coef_
            vocab = _vectorizer.vocabulary_
            
            word_scores = []
            for word in words:
                score = 0.0
                if word in vocab:
                    idx = vocab[word]
                    if len(classes) == 2:
                        score = float(coef[0][idx]) if classes[1] == "positive" else -float(coef[0][idx])
                    else:
                        p_w = coef[pos_idx][idx] if pos_idx != -1 else 0.0
                        n_w = coef[neg_idx][idx] if neg_idx != -1 else 0.0
                        score = float(p_w - n_w)
                word_scores.append((word, score))
            return word_scores

    # For hybrid model, tokenize and score with lexicon
    tokens = tokenize(cleaned)
    pos_keys = sorted(POSITIVE_TERMS.keys(), key=len, reverse=True)
    neg_keys = sorted(NEGATIVE_TERMS.keys(), key=len, reverse=True)
    
    word_scores = []
    for token in tokens:
        score = 0.0
        if token in POSITIVE_TERMS:
            score = POSITIVE_TERMS[token]
        elif token in NEGATIVE_TERMS:
            score = NEGATIVE_TERMS[token]
        else:
            for pk in pos_keys:
                if token.startswith(pk):
                    score = POSITIVE_TERMS[pk]
                    break
            if not score:
                for nk in neg_keys:
                    if token.startswith(nk):
                        score = NEGATIVE_TERMS[nk]
                        break
        word_scores.append((token, score))
    return word_scores

