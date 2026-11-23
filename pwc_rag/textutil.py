import re

# Do not split after common legal/financial abbreviations ("Rs. 20 crore", "No. 5", "Sec. 4", "Ltd.", "Pvt.").
_SENT = re.compile(r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bSec\.)(?<!\bLtd\.)(?<!\bPvt\.)(?<!\bArt\.)(?<!\bvs\.)(?<=[.;])\s+|\n{2,}")
# A single newline is a hard wrap (typical in PDFs) unless the next line starts a new clause/list item.
_WRAP = re.compile(r"(?<!\n)\n(?!\n|\s*(?:Section|Clause|Rule|Regulation|Article)\s+\d|\s*\d+(?:\.\d+)*[.)]\s|\s*[-*•#])")


def split_sentences(text: str) -> list[str]:
    text = _WRAP.sub(" ", text)
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]
