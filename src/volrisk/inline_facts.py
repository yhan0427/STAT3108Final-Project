"""Limited original-filing USD facts for spot checks, not a conformant XBRL processor."""
from decimal import Decimal, InvalidOperation
import re

from bs4 import BeautifulSoup


def extract_usd_facts(html, cik):
    soup = BeautifulSoup(html, "lxml")
    contexts = {}
    for node in soup.find_all(lambda x: x.name and x.name.lower() == "xbrli:context"):
        if node.find(lambda x: x.name and x.name.lower() in {"xbrldi:explicitmember", "xbrldi:typedmember"}):
            continue
        identifier = node.find(lambda x: x.name and x.name.lower() == "xbrli:identifier")
        if identifier is None or not identifier.get_text(strip=True).isdigit() or int(identifier.get_text(strip=True)) != int(cik):
            continue
        def date(tag):
            value = node.find(lambda x: x.name and x.name.lower() == f"xbrli:{tag}")
            return value.get_text(strip=True) if value else ""
        contexts[node.get("id")] = {"start": date("startdate"), "end": date("instant") or date("enddate")}
    units = {}
    for node in soup.find_all(lambda x: x.name and x.name.lower() == "xbrli:unit"):
        measures = node.find_all(lambda x: x.name and x.name.lower() == "xbrli:measure")
        if len(measures) == 1 and measures[0].get_text(strip=True).lower() == "iso4217:usd":
            units[node.get("id")] = "USD"
    facts = []
    for node in soup.find_all(lambda x: x.name and x.name.lower() == "ix:nonfraction"):
        name, context = node.get("name", ""), contexts.get(node.get("contextref"))
        if not name.startswith("us-gaap:") or context is None or node.get("unitref") not in units:
            continue
        if str(node.get("xsi:nil", "")).lower() in {"true", "1"}:
            continue
        transform = node.get("format", "").split(":")[-1].lower()
        if transform not in {"", "num-dot-decimal", "numdotdecimal", "numcommadot", "zerodash", "zero-dash"}:
            continue  # Never guess unsupported transformations.
        raw = "".join(str(s) for s in node.strings if not any(
            getattr(p, "name", "").lower() == "ix:exclude" for p in s.parents))
        raw = re.sub(r"[\s,]", "", raw)
        if transform in {"zerodash", "zero-dash"} and raw in {"-", "—", "–"}:
            raw = "0"
        try:
            value = Decimal(raw)
            if not value.is_finite() or value < 0:
                continue  # Inline numerics use the sign attribute, not a guessed display sign.
            value *= Decimal(10) ** int(node.get("scale", "0"))
            if node.get("sign") == "-":
                value = -value
        except (InvalidOperation, ValueError):
            continue
        facts.append({"tag": name.split(":", 1)[1], **context, "value": float(value),
                      "context_id": node.get("contextref"), "fact_id": node.get("id", ""),
                      "scale": int(node.get("scale", "0"))})
    return facts
