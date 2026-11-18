from volrisk.mda import extract_mda, visible_text


def test_toc_and_cross_reference_do_not_replace_actual_section():
    title = "Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations"
    end = "Item 7A. Quantitative and Qualitative Disclosures about Market Risk"
    body = "Operating revenues increased while costs rose. " * 100
    html = f"<html><p>{title} 20 {end} 40</p><p>See Part II, {title} for details.</p><p>{title}</p><p>{body}</p><p>{end}</p></html>"
    result = extract_mda(html)
    assert result["status"] == "candidate_needs_review"
    assert result["text"] == title + " " + body.strip()
    assert result["review_required"]


def test_missing_end_is_rejected_and_hidden_text_removed():
    html = "<!DOCTYPE html><html><!--COMMENT--><ix:hidden>SECRET</ix:hidden><p style='display: none'>HIDDEN</p><script>CODE</script><p>Visible</p></html>"
    assert visible_text(html) == "Visible"
    result = extract_mda("Item 7. Management's Discussion and Analysis " + "body " * 1000)
    assert result["status"] == "rejected_no_plausible_boundary"
    assert result["text"] == ""


def test_inline_small_caps_and_headings_split_across_blocks():
    body = "Cash flows and operating results require discussion. " * 100
    html = ("<html><div><span>I</span><span>TEM</span> 7.</div>"
            "<div>MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS.</div>"
            f"<p>{body}</p><div>ITEM 7A.</div>"
            "<div>QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK.</div></html>")
    result = extract_mda(html)
    assert result["status"] == "candidate_needs_review"
    assert result["text"].startswith("ITEM 7. MANAGEMENT’S")
    assert result["text"].endswith(body.strip())


def test_quoted_title_and_internal_item7a_reference_are_not_boundaries():
    title = "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    end = "Item 7A. Quantitative and Qualitative Disclosures About Market Risk"
    body = "This discusses our operating results. " * 100
    html = f"<p>{title}” and the following material describe risks.</p><p>{title}</p><p>{body}</p><p>See {end} for related information.</p><p>{body}</p><p>{end}</p>"
    result = extract_mda(html)
    assert result["text"].startswith(title + " This discusses")
    assert f"See {end}" in result["text"]


def test_part_ii_prefix_singular_disclosure_and_optional_heading_comma():
    body = "Operating income and cash balances changed. " * 100
    html = ("<p>Part II. Item 7. Management's Discussion and Analysis of Financial Condition, and Results of Operations</p>"
            f"<p>{body}</p><p>Part II. Item 7(a). Quantitative and Qualitative Disclosure About Market Risk</p>")
    result = extract_mda(html)
    assert result["status"] == "candidate_needs_review"
    assert result["end_heading"].startswith("Item 7(a)")
    assert result["text"].endswith(body.strip() + " Part II.")
