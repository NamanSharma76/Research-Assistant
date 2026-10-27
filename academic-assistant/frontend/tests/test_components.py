"""Unit tests for frontend UI components and formatters."""
import unittest
from components import (
    render_citations_to_html,
    format_trace,
    render_status_badge,
    outline_to_markdown,
    outline_to_text,
    render_error_message,
)
from api_client import ApiError


class TestComponents(unittest.TestCase):
    def test_single_citation(self):
        text = "This is a key claim [Introduction, p. 2]."
        html_out = render_citations_to_html(text)
        self.assertIn("class=\"citation-chip\"", html_out)
        self.assertIn("Introduction, p. 2", html_out)
        self.assertIn(">[Introduction, p. 2]</span>", html_out)

    def test_page_range_citation_hyphen(self):
        text = "See methodology [Methods, pp. 4-5]."
        html_out = render_citations_to_html(text)
        self.assertIn("class=\"citation-chip\"", html_out)
        self.assertIn("Methods, pp. 4-5", html_out)

    def test_page_range_citation_endash(self):
        text = "See results [Results, pp. 10–12]."
        html_out = render_citations_to_html(text)
        self.assertIn("class=\"citation-chip\"", html_out)
        self.assertIn("Results, pp. 10–12", html_out)

    def test_multiple_citations(self):
        text = "First claim [Abstract, p. 1] and second [Conclusion, p. 8]."
        html_out = render_citations_to_html(text)
        self.assertEqual(html_out.count("citation-chip"), 2)
        self.assertIn("Abstract, p. 1", html_out)
        self.assertIn("Conclusion, p. 8", html_out)

    def test_no_citations(self):
        text = "No citations here, just plain text."
        html_out = render_citations_to_html(text)
        self.assertEqual(html_out, text)

    def test_math_brackets_untouched(self):
        text = "Consider vectors [x, y] and intervals [0, 1] or array [1, 2, 3]."
        html_out = render_citations_to_html(text)
        self.assertNotIn("citation-chip", html_out)
        self.assertIn("[x, y]", html_out)
        self.assertIn("[0, 1]", html_out)
        self.assertIn("[1, 2, 3]", html_out)

    def test_html_escaping(self):
        text = "<script>alert('xss')</script> [Security, p. 99]"
        html_out = render_citations_to_html(text)
        self.assertNotIn("<script>", html_out)
        self.assertIn("&lt;script&gt;", html_out)
        self.assertIn("class=\"citation-chip\"", html_out)
        self.assertIn("Security, p. 99", html_out)

    def test_format_trace_list(self):
        trace = ["router", "retrieve", "answer"]
        formatted = format_trace(trace)
        self.assertEqual(formatted, "router ➔ retrieve ➔ answer")

    def test_format_trace_string(self):
        trace = "router, retrieve, answer"
        formatted = format_trace(trace)
        self.assertEqual(formatted, "router, retrieve, answer")

    def test_format_trace_none_empty(self):
        self.assertEqual(format_trace(None), "N/A")
        self.assertEqual(format_trace([]), "N/A")
        self.assertEqual(format_trace(""), "N/A")

    def test_render_status_badge(self):
        self.assertIn("Ready", render_status_badge("ready"))
        self.assertIn("Processing", render_status_badge("processing"))
        self.assertIn("Failed", render_status_badge("failed"))
        self.assertIn("other", render_status_badge("other"))

    def test_outline_to_markdown(self):
        outline_data = {
            "title": "Attention Mechanism",
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Overview",
                    "bullets": ["Point A", "Point B"],
                    "source": "[Intro, p. 1]",
                }
            ],
        }
        md = outline_to_markdown(outline_data)
        self.assertIn("# Attention Mechanism", md)
        self.assertIn("## Slide 1: Overview", md)
        self.assertIn("- Point A", md)
        self.assertIn("- Point B", md)
        self.assertIn("**Sources:** [Intro, p. 1]", md)

    def test_outline_to_text(self):
        outline_data = {
            "title": "Attention Mechanism",
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Overview",
                    "bullets": ["Point A", "Point B"],
                    "source": "[Intro, p. 1]",
                }
            ],
        }
        txt = outline_to_text(outline_data)
        self.assertIn("ATTENTION MECHANISM", txt)
        self.assertIn("Slide 1: Overview", txt)
        self.assertIn("  * Point A", txt)
        self.assertIn("  * Point B", txt)
        self.assertIn("Sources: [Intro, p. 1]", txt)

    def test_render_error_message(self):
        err401 = ApiError(401, "Session expired")
        self.assertIn("Session expired", render_error_message(err401))

        err409 = ApiError(409, "Conflict")
        self.assertIn("still being processed", render_error_message(err409))

        err429 = ApiError(429, "Rate limited", retry_after=15)
        self.assertIn("Too many messages, try again in 15 seconds.", render_error_message(err429))

        # Real 502/503 from backend (AI provider error)
        err503_provider = ApiError(503, "The AI provider is temporarily unavailable, please retry.", is_connection_error=False)
        self.assertEqual("The AI provider is temporarily unavailable, please retry.", render_error_message(err503_provider))

        # Backend offline (connection error)
        err_offline = ApiError(503, "Backend not reachable. Check that the server is running", is_connection_error=True)
        self.assertEqual("Backend not reachable. Check that the server is running", render_error_message(err_offline))

        # Backend offline (timeout)
        err_timeout = ApiError(504, "Backend not reachable. Check that the server is running", is_connection_error=True)
        self.assertEqual("Backend not reachable. Check that the server is running", render_error_message(err_timeout))


if __name__ == "__main__":
    unittest.main()
