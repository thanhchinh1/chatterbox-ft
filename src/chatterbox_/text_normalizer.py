import re
from typing import Dict

from num2words import num2words

from .g2p_vi import g2p_vi


_ABBREVIATIONS: Dict[str, str] = {
    "TP": "thanh pho",
    "Q": "quan",
    "P": "phuong",
    "HCM": "ho chi minh",
    "HN": "ha noi",
    "VN": "viet nam",
}

_UNIT_MAP: Dict[str, str] = {
    "kg": "ki lo gam",
    "g": "gam",
    "mg": "mi li gam",
    "km": "ki lo met",
    "m": "met",
    "cm": "xen ti met",
    "mm": "mi li met",
    "m2": "met vuong",
    "m3": "met khoi",
    "%": "phan tram",
    "đ": "dong",
    "₫": "dong",
}

_DECIMAL_RE = re.compile(r"(\d+)[,.](\d+)")
_INT_RE = re.compile(r"\d+")


def _expand_decimal(match: re.Match) -> str:
    whole, frac = match.group(1), match.group(2)
    try:
        whole_words = num2words(int(whole), lang="vi")
        frac_words = " ".join(num2words(int(d), lang="vi") for d in frac)
        return f"{whole_words} phẩy {frac_words}"
    except Exception:
        return match.group(0)


def _expand_int(match: re.Match) -> str:
    try:
        return num2words(int(match.group(0)), lang="vi")
    except Exception:
        return match.group(0)


def _expand_units(text: str) -> str:
    for unit, spoken in _UNIT_MAP.items():
        text = re.sub(rf"(\d+)\s*{re.escape(unit)}\b", rf"\1 {spoken}", text, flags=re.IGNORECASE)
    return text


def _expand_abbreviations(text: str) -> str:
    for abbr, spoken in _ABBREVIATIONS.items():
        text = re.sub(rf"\b{abbr}\.\b", spoken, text)
        text = re.sub(rf"\b{abbr}\b", spoken, text)
    return text


def normalize_vi_text(
    text: str,
    *,
    use_phoneme: bool = True,
    use_g2p: bool = True,
    expand_numbers: bool = True,
    expand_abbrev: bool = True,
) -> str:
    """
    Vietnamese text normalization optimized for grapheme tokenizer.
    The phoneme path is intentionally light-weight and keeps characters in-vocab.
    """
    if not text:
        return text

    text = text.strip()

    if expand_abbrev:
        text = _expand_abbreviations(text)

    if expand_numbers:
        text = _expand_units(text)
        text = _DECIMAL_RE.sub(_expand_decimal, text)
        text = _INT_RE.sub(_expand_int, text)

    if use_phoneme:
        if use_g2p:
            g2p_text = g2p_vi(text)
            if g2p_text:
                text = g2p_text
        text = re.sub(r"\s+", " ", text)

    return text
