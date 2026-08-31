import unittest

from ingest import extract_sections, make_chunks, split_words


class IngestTests(unittest.TestCase):
    def test_extracts_heading_paths_and_table(self):
        markup = b"""<html><head><link rel="canonical" href="https://example.test/x"></head>
        <body><nav>Ignore me</nav><main><h1>Housing grant</h1><p>Introduction.</p>
        <h2>Eligibility</h2><p>Applicants must qualify.</p>
        <table><tr><th>Type</th><th>Amount</th></tr><tr><td>Grant</td><td>$1</td></tr></table>
        </main></body></html>"""
        canonical, sections = extract_sections(markup)
        self.assertEqual(canonical, "https://example.test/x")
        self.assertEqual(sections[0].heading_path, ("Housing grant",))
        self.assertEqual(sections[1].heading_path, ("Housing grant", "Eligibility"))
        self.assertIn("Type | Amount", sections[1].text)

    def test_chunk_overlap(self):
        chunks = list(split_words("one two three four five six", 4, 2))
        self.assertEqual(chunks, ["one two three four", "three four five six"])

    def test_extracts_next_data_embedded_article(self):
        payload = {
            "props": {"pageProps": {"layoutData": {
                "bodyContent": {"value": "<h2>Eligibility</h2><p>Full article copy.</p>"}
            }}}
        }
        markup = (
            '<html><body><main><h1>Shell</h1></main>'
            f'<script id="__NEXT_DATA__" type="application/json">{__import__("json").dumps(payload)}</script>'
            '</body></html>'
        ).encode()
        _, sections = extract_sections(markup)
        self.assertEqual(sections[0].heading_path, ("Eligibility",))
        self.assertEqual(sections[0].text, "Full article copy.")

    def test_extracts_salesforce_rich_text_leaf_divs(self):
        markup = b"""<main><div class="faq"><span class="question">Rates?</span>
        <lightning-formatted-rich-text><span><div>Rate period.</div>
        <table><tr><th>Account</th><th>Rate</th></tr><tr><td>OA</td><td>2.5%</td></tr></table>
        <div>More information.</div></span></lightning-formatted-rich-text></div></main>"""
        _, sections = extract_sections(markup)
        self.assertEqual(len(sections), 1)
        self.assertIn("Rate period.", sections[0].text)
        self.assertIn("Account | Rate", sections[0].text)
        self.assertIn("More information.", sections[0].text)

    def test_chunk_metadata(self):
        source = {
            "source_id": "X01", "agency": "Agency", "title": "Title",
            "url": "https://example.test/x", "domain": "test", "priority": "core",
            "retrieved_at": "2026-01-01",
        }
        _, sections = extract_sections(b"<main><h1>Title</h1><p>alpha beta gamma</p></main>")
        chunks = make_chunks(source, None, sections, 2, 0, "abc")
        self.assertEqual(chunks[0]["chunk_id"], "X01:000:000")
        self.assertEqual(chunks[0]["metadata"]["heading_path"], ["Title"])


if __name__ == "__main__":
    unittest.main()
