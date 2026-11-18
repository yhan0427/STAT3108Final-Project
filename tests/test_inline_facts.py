from volrisk.inline_facts import extract_usd_facts


def test_scale_sign_currency_and_consolidated_context_are_explicit():
    html = '''<html><xbrli:context id="base"><xbrli:entity><xbrli:identifier>0000000001</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:context id="segment"><xbrli:entity><xbrli:identifier>0000000001</xbrli:identifier><xbrli:segment><xbrldi:explicitmember>Segment</xbrldi:explicitmember></xbrli:segment></xbrli:entity><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>
    <xbrli:unit id="usd"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit><xbrli:unit id="eur"><xbrli:measure>iso4217:EUR</xbrli:measure></xbrli:unit>
    <ix:nonfraction name="us-gaap:Assets" unitref="usd" contextref="base" scale="6" format="ixt:num-dot-decimal">1,234</ix:nonfraction>
    <ix:nonfraction name="us-gaap:StockholdersEquity" unitref="usd" contextref="base" scale="3" sign="-">12</ix:nonfraction>
    <ix:nonfraction name="us-gaap:Assets" unitref="usd" contextref="segment">9</ix:nonfraction>
    <ix:nonfraction name="us-gaap:Assets" unitref="eur" contextref="base">10</ix:nonfraction></html>'''
    rows = extract_usd_facts(html, 1)
    assert len(rows) == 2
    assert rows[0]["value"] == 1234000000
    assert rows[1]["value"] == -12000
    assert rows[0]["start"] == ""
    assert not extract_usd_facts(html, 2)
