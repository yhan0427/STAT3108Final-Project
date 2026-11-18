"""Conservative Item 7 candidates with explicit provenance and review requirements."""
import re
from bisect import bisect_right
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag, Comment, Doctype, Declaration, ProcessingInstruction

START = re.compile(
    r"\bitem\s*7\s*[.\-:–—|]?\s*management[’'`]?s\s+discussion\s+and\s+analysis"
    r"(?:\s+of\s+(?:financial\s+condition,?\s+and\s+results\s+of\s+operations|"
    r"results\s+of\s+operations\s+and\s+financial\s+condition))?", re.I)
END = re.compile(
    r"\bitem\s*(?:7\s*\(?\s*a\s*\)?\s*[.\-:–—|]?\s*quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market(?:\s+and\s+business)?\s+risks?|"
    r"8\s*[.\-:–—|]?\s*(?:consolidated\s+)?financial\s+statements(?:\s+and\s+supplementary\s+data)?)", re.I)
REFERENCE = re.compile(
    r"(?:see|refer(?:red)?\s+to|in|under|within|of)\s+(?:part\s+ii\s*[,.:]?\s*)?[\"'“‘(\s]*$", re.I)


BLOCKS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "td", "th", "tr", "li", "section", "article", "table"}


def _clean(html):
    soup = BeautifulSoup(html, "lxml")
    for node in list(soup.find_all(True)):
        if node.parent is None:
            continue
        tag = node.name.lower()
        style = re.sub(r"\s+", "", str(node.get("style", "")).lower())
        if tag in {"script", "style", "noscript", "ix:hidden"} or node.has_attr("hidden") or "display:none" in style or "visibility:hidden" in style:
            node.decompose()
    return soup


def _render(root, trace=False):
    chunks, spans = [], []
    offset = 0
    pending_space = False
    def walk(node, block=None):
        nonlocal offset, pending_space
        if isinstance(node, Tag):
            is_block = node.name.lower() in BLOCKS or node.name.lower() == "br"
            if is_block:
                pending_space = offset > 0
            for child in node.children:
                walk(child, node if node.name.lower() in BLOCKS else block)
            if is_block:
                pending_space = offset > 0
        elif isinstance(node, (Comment, Doctype, Declaration, ProcessingInstruction)):
            return
        elif isinstance(node, NavigableString):
            # Preserve inline small-cap fragments: <span>I</span><span>TEM</span>
            # is ITEM, whereas a paragraph boundary remains whitespace.
            for part in re.findall(r"\S+|\s+", unicodedata.normalize("NFKC", str(node))):
                if part.isspace():
                    pending_space = offset > 0
                else:
                    if pending_space:
                        chunks.append(" ")
                        offset += 1
                        pending_space = False
                    start = offset
                    chunks.append(part)
                    offset += len(part)
                    if trace:
                        spans.append((start, offset, block))
    walk(root)
    return "".join(chunks).strip(), spans


def visible_text(html):
    return _render(_clean(html))[0]


def _structural_heading(match, pattern, spans, offsets):
    index = bisect_right(offsets, match.start()) - 1
    if index < 0:
        return False
    block = spans[index][2]
    end_index = bisect_right(offsets, match.end() - 1) - 1
    end_block = spans[end_index][2]
    if block is not None and end_block is not None and block is not end_block:
        first, last = _render(block)[0], _render(end_block)[0]
        if len(first) <= 600 and len(last) <= 600:
            if match.group().startswith(first) and match.group().endswith(re.sub(r"[\d\s.,:;—–-]+$", "", last)):
                return True
    for _ in range(4):
        if block is None:
            break
        block_text = _render(block)[0]
        if len(block_text) <= 600:
            found = pattern.search(block_text)
            if found and re.fullmatch(r"(?:part\s+ii[.,:]?\s*)?[\d\s().]*", block_text[:found.start()], re.I):
                # Quoted cross-references, subtitles and prose after the heading
                # do not qualify. Page numbers in table-of-contents rows may.
                if re.fullmatch(r"[\d\s.,:;—–-]*", block_text[found.end():]):
                    return True
        block = next((p for p in block.parents if getattr(p, "name", None) in BLOCKS), None)
    return False


def _reference(text, match):
    return bool(REFERENCE.search(text[max(0, match.start() - 100):match.start()]))


def extract_mda(html, min_chars=2500, max_chars=350000):
    text, spans = _render(_clean(html), trace=True)
    offsets = [s[0] for s in spans]
    mentions = list(START.finditer(text))
    starts = [m for m in mentions if _structural_heading(m, START, spans, offsets)]
    ends = [m for m in END.finditer(text) if _structural_heading(m, END, spans, offsets)]
    candidates = []
    for start in starts:
        following = [end for end in ends if end.start() > start.end()]
        if not following:
            continue
        end = following[0]
        size = end.start() - start.start()
        if min_chars <= size <= max_chars:
            candidates.append((start, end))
    base = {"visible_document_chars": len(text), "item7_mentions": len(mentions), "structural_start_headings": len(starts),
            "body_candidate_count": len(candidates), "review_required": True}
    if not candidates:
        return {**base, "status": "rejected_no_plausible_boundary", "text": ""}
    # Repeated page headers are kept inside the longest plausible section. Directory
    # spans and explicit cross-references are excluded before ranking; no outcome
    # or company-specific offset participates. Every pilot still requires review.
    start, end = max(candidates, key=lambda pair: pair[1].start() - pair[0].start())
    section = text[start.start():end.start()].strip()
    return {**base, "status": "candidate_needs_review", "text": section,
            "start_char": start.start(), "end_char": end.start(),
            "section_chars": len(section), "section_words": len(section.split()),
            "start_heading": start.group(), "end_heading": end.group(),
            "before_start": text[max(0, start.start() - 200):start.start()],
            "start_preview": text[start.start():start.start() + 600],
            "end_preview": text[max(start.start(), end.start() - 600):end.start() + 250]}
