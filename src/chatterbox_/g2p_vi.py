import re
from typing import Optional


_ALLOWED_RE = re.compile(
    r"[^a-z0-9 àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ.,'!?-]"
)

try:
    from viphoneme import vi2IPA  # type: ignore

    _VI2IPA = vi2IPA
except Exception:
    _VI2IPA = None


def is_g2p_available() -> bool:
    return _VI2IPA is not None


_IPA_REPLACEMENTS = [
    ("tʰ", "th"),
    ("kʰ", "kh"),
    ("pʰ", "ph"),
    ("cʰ", "ch"),
    ("ɯ", "ư"),
    ("ɤ", "ơ"),
    ("ə", "â"),
    ("ɐ", "â"),
    ("ɔ", "o"),
    ("ɒ", "o"),
    ("ɪ", "i"),
    ("ʊ", "u"),
    ("ʌ", "â"),
    ("ŋ", "ng"),
    ("ɲ", "nh"),
    ("ʂ", "s"),
    ("ʃ", "s"),
    ("ʈ", "t"),
    ("ʒ", "d"),
    ("ɡ", "g"),
    ("ʔ", ""),
    ("ʰ", ""),
    ("ˈ", ""),
    ("ˌ", ""),
    ("ː", ""),
    ("ˑ", ""),
    ("͡", ""),
    ("̃", ""),
    ("̈", ""),
]


def _ipa_to_vi_text(text: str) -> str:
    text = text.replace("_", " ")
    for src, dst in _IPA_REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def g2p_vi(text: str) -> Optional[str]:
    if _VI2IPA is None:
        return None

    try:
        out = _VI2IPA(text)
    except Exception:
        return None

    out = str(out)
    out = _ipa_to_vi_text(out)

    out = out.lower().strip()
    out = _ALLOWED_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()

    return out or None
