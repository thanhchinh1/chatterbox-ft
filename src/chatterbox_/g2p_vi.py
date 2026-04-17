import re
from typing import Optional


_ALLOWED_RE = re.compile(r"[^a-z0-9 .,'-]")

try:
    from g2pV import G2p  # type: ignore

    _G2P = G2p()
except Exception:
    _G2P = None


def is_g2p_available() -> bool:
    return _G2P is not None


def g2p_vi(text: str) -> Optional[str]:
    if _G2P is None:
        return None

    try:
        out = _G2P(text)
    except Exception:
        return None

    if isinstance(out, list):
        out = " ".join(str(tok) for tok in out)
    else:
        out = str(out)

    out = out.lower().strip()
    out = _ALLOWED_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()

    return out or None
