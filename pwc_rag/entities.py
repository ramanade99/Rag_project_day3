"""STEP 2/3 - Entity-relation extraction (LLM-guided, rule-based fallback) and entity normalisation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import LLM
from .textutil import split_sentences

# Canonical entity -> aliases. Normalisation is what lets the graph link "Income Tax Dept" to "CBDT".
ALIASES: dict[str, list[str]] = {
    "CBDT": ["CBDT", "Central Board of Direct Taxes", "Income Tax Dept", "Income Tax Department", "Tax Authority"],
    "CBIC": ["CBIC", "Central Board of Indirect Taxes and Customs", "Central Board of Indirect Taxes"],
    "RBI": ["RBI", "Reserve Bank of India", "Reserve Bank"],
    "SEBI": ["SEBI", "Securities and Exchange Board of India"],
    "GST Council": ["GST Council", "Goods and Services Tax Council"],
    "FIU-IND": ["FIU-IND", "Financial Intelligence Unit", "FIU"],
    "Income Tax Act": ["Income Tax Act", "Income-tax Act", "IT Act", "Income Tax Act, 1961"],
    "CGST Act": ["CGST Act", "GST Act", "Central Goods and Services Tax Act"],
    "PMLA": ["PMLA", "Prevention of Money Laundering Act", "Prevention of Money-Laundering Act"],
    "Companies Act 2013": ["Companies Act 2013", "Companies Act, 2013", "Companies Act"],
    "FEMA": ["FEMA", "Foreign Exchange Management Act"],
    "FCRA": ["FCRA", "Foreign Contribution (Regulation) Act"],
    "KYC": ["KYC", "Know Your Customer"],
    "E-Invoicing": ["e-invoicing", "e-invoice", "einvoicing", "electronic invoicing"],
    "ISD": ["ISD", "Input Service Distributor"],
    "TDS": ["TDS", "Tax Deducted at Source"],
    "Material Adverse Change": ["Material Adverse Change", "MAC", "Material Adverse Effect"],
    "Indemnity": ["indemnity", "indemnities", "indemnification", "indemnify"],
    "Share Purchase Agreement": ["Share Purchase Agreement", "SPA"],
    "BRSR": ["BRSR", "Business Responsibility and Sustainability Report"],
    "ISO 31000": ["ISO 31000"],
    "Internal Audit": ["internal audit", "internal audit controls"],
    "Beneficial Owner": ["beneficial owner", "beneficial ownership"],
    "Enhanced Due Diligence": ["enhanced due diligence", "EDD"],
}
_ALIAS_INDEX = {a.lower(): canon for canon, al in ALIASES.items() for a in al}
_ALIAS_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(a) for a in sorted(_ALIAS_INDEX, key=len, reverse=True)) + r")(?![\w-])", re.I)
_REF_RE = re.compile(r"\b(Section|Rule|Clause|Regulation|Article)\s+(\d+[A-Z]{0,3}(?:\(\w+\))*)", re.I)

# Relation verbs (surface form -> normalised relation label)
RELATIONS = {
    r"\bissued? by\b": "issued_by", r"\bissues?\b": "issues", r"\bamend(?:s|ed)?\b": "amends",
    r"\bgovern(?:s|ed by)?\b": "governs", r"\brequires?\b|\bmandates?\b": "requires",
    r"\bpenali[sz]es?\b|\bpenalty\b": "penalises", r"\bsupersedes?\b": "supersedes",
    r"\bapplies to\b|\bapplicable to\b": "applies_to", r"\bdefined (?:in|under)\b": "defined_in",
    r"\bprescribes?\b": "prescribes", r"\bnotif(?:ies|ied)\b": "notifies", r"\bclarif(?:ies|ied)\b": "clarifies",
    r"\bunder\b": "under", r"\bexempts?\b": "exempts", r"\bcovers?\b": "covers", r"\bexceeds?\b": "exceeds",
}


def normalize_entity(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().strip(".,;:"))
    canon = _ALIAS_INDEX.get(n.lower())
    if canon:
        return canon
    m = _REF_RE.fullmatch(n)
    return f"{m.group(1).title()} {m.group(2).upper()}" if m else n


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    chunk_id: str = ""


def find_entities(text: str) -> list[tuple[str, int, int]]:
    spans = [(normalize_entity(m.group(0)), m.start(), m.end()) for m in _ALIAS_RE.finditer(text)]
    for m in _REF_RE.finditer(text):
        at_line_start = not text[:m.start()].strip() or text[:m.start()].rstrip(" \t").endswith("\n")
        if at_line_start and text[m.end():m.end() + 1] in (".", ":"):
            continue  # structural heading ("Section 1. Penalty"), not a statutory reference
        spans.append((normalize_entity(m.group(0)), m.start(), m.end()))
    return sorted(spans, key=lambda s: s[1])


def extract_rule_based(text: str, chunk_id: str = "") -> list[Triple]:
    out: list[Triple] = []
    for sent in split_sentences(text):
        ents = find_entities(sent)
        for (a, _, ae), (b, bs, _) in zip(ents, ents[1:]):
            if a == b:
                continue
            between = sent[ae:bs]
            rel = next((lab for pat, lab in RELATIONS.items() if re.search(pat, between, re.I)), "related_to")
            out.append(Triple(a, rel, b, chunk_id))
    # Paragraph-level co-occurrence: link entities that share a paragraph but no sentence (weaker evidence).
    linked = {frozenset((t.head, t.tail)) for t in out}
    for para in re.split(r"\n\s*\n|\n(?=\s*(?:Section|Clause|Rule)\s+\d)", text):
        names = list(dict.fromkeys(n for n, *_ in find_entities(para)))
        for a, b in zip(names, names[1:]):
            if frozenset((a, b)) not in linked:
                out.append(Triple(a, "co_occurs_with", b, chunk_id))
                linked.add(frozenset((a, b)))
    return out


_PROMPT = """EXTRACT_TRIPLES
Extract regulatory entity-relation triples from the text. Return ONLY a JSON list of
[head, relation, tail] with short canonical names (regulators, statutes, section/rule references, obligations).
TEXT:
{text}"""


def extract_triples(text: str, chunk_id: str = "", llm: LLM | None = None) -> list[Triple]:
    """LLM-guided extraction merged with rule-based extraction; every name is normalised."""
    triples = extract_rule_based(text, chunk_id)
    if llm is not None and llm.name != "offline-extractive":
        try:
            raw = llm.invoke("You extract knowledge-graph triples. Output JSON only.", _PROMPT.format(text=text[:4000]))
            data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
            for item in data:
                if isinstance(item, list) and len(item) == 3 and all(isinstance(x, str) and x.strip() for x in item):
                    h, r, t = normalize_entity(item[0]), re.sub(r"\W+", "_", item[1].strip().lower()), normalize_entity(item[2])
                    if h != t:
                        triples.append(Triple(h, r, t, chunk_id))
        except Exception:
            pass  # rule-based triples still stand
    return list(dict.fromkeys(triples))
