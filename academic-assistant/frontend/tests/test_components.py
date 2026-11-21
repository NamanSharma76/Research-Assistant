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

    def test_outline_to_markdown_string_slides(self):
        outline_data = {
            "title": "Presentation Outline",
            "slides": [
                "Overview of Transformer architecture",
                "Empirical evaluation on WMT 2014",
            ],
        }
        md = outline_to_markdown(outline_data)
        self.assertIn("# Presentation Outline", md)
        self.assertIn("## Slide 1: Slide 1", md)
        self.assertIn("- Overview of Transformer architecture", md)
        self.assertIn("## Slide 2: Slide 2", md)
        self.assertIn("- Empirical evaluation on WMT 2014", md)

    def test_outline_to_markdown_mixed_slides_and_string_bullets(self):
        outline_data = {
            "title": "Mixed Deck",
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Introduction",
                    "bullets": ["Point A", "Point B"],
                    "source": "[Intro, p. 1]",
                },
                {
                    "slide_number": 2,
                    "title": "Methodology",
                    "bullets": "Single string bullet should not split chars",
                },
                "Bare string slide content",
            ],
        }
        md = outline_to_markdown(outline_data)
        self.assertIn("## Slide 1: Introduction", md)
        self.assertIn("- Point A", md)
        self.assertIn("## Slide 2: Methodology", md)
        self.assertIn("- Single string bullet should not split chars", md)
        self.assertNotIn("- S\n- i\n- n", md)
        self.assertIn("## Slide 3: Slide 3", md)
        self.assertIn("- Bare string slide content", md)

    def test_outline_to_text_string_slides(self):
        outline_data = {
            "title": "Presentation Text",
            "slides": [
                "Overview slide string",
                "Conclusion slide string",
            ],
        }
        txt = outline_to_text(outline_data)
        self.assertIn("PRESENTATION TEXT", txt)
        self.assertIn("Slide 1: Slide 1", txt)
        self.assertIn("  * Overview slide string", txt)
        self.assertIn("Slide 2: Slide 2", txt)
        self.assertIn("  * Conclusion slide string", txt)

    def test_outline_to_text_mixed_slides_and_string_bullets(self):
        outline_data = {
            "title": "Mixed Text Deck",
            "slides": [
                {
                    "slide_number": 1,
                    "title": "Overview",
                    "bullets": ["Dict bullet 1"],
                },
                {
                    "slide_number": 2,
                    "title": "Details",
                    "bullets": "Single string bullet in text",
                },
                "String slide in text",
            ],
        }
        txt = outline_to_text(outline_data)
        self.assertIn("Slide 1: Overview", txt)
        self.assertIn("  * Dict bullet 1", txt)
        self.assertIn("Slide 2: Details", txt)
        self.assertIn("  * Single string bullet in text", txt)
        self.assertNotIn("  * S\n  * i", txt)
        self.assertIn("Slide 3: Slide 3", txt)
        self.assertIn("  * String slide in text", txt)

    def test_notes_view_key_terms_dict_and_string_normalization(self):
        from unittest.mock import MagicMock, patch
        import views.notes as notes_view

        mock_st = MagicMock()
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        c1.button.return_value = False
        c2.button.return_value = False
        mock_st.columns.return_value = (c1, c2, c3)
        mock_st.session_state = {
            "notes_test_doc": {
                "key_terms": [
                    {
                        "term": "Self-Attention",
                        "definition": "Relates different positions [Architecture, p. 3]",
                        "sources": ["p. 3"],
                    },
                    "LayerNorm [Optimization, pp. 4-5]",
                    "<script>alert('xss')</script> [Sec 1, p. 1]",
                ]
            }
        }

        with patch("views.notes.st", mock_st):
            notes_view.render_notes_view("fake_token", "test_doc")

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        joined_output = "\n".join(markdown_calls)

        self.assertIn("**Self-Attention**", joined_output)
        self.assertIn("Relates different positions", joined_output)
        self.assertIn("Architecture, p. 3", joined_output)
        self.assertIn("citation-chip", joined_output)
        self.assertIn("LayerNorm", joined_output)
        self.assertIn("Optimization, pp. 4-5", joined_output)
        self.assertNotIn("<script>", joined_output)
        self.assertIn("&lt;script&gt;", joined_output)

    def test_notes_view_revision_questions_dict_and_string_normalization(self):
        from unittest.mock import MagicMock, patch
        import views.notes as notes_view

        mock_st = MagicMock()
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        c1.button.return_value = False
        c2.button.return_value = False
        mock_st.columns.return_value = (c1, c2, c3)
        mock_st.session_state = {
            "notes_test_doc": {
                "revision_questions": [
                    {
                        "question": "What is Q? [Model, p. 3]",
                        "expected_answer": "Query vector",
                        "sources": ["p. 3"],
                    },
                    {
                        "question": "What is K?",
                        "answer": "Key vector fallback",
                    },
                    "How is BLEU evaluated? [Results, pp. 6-7]",
                    "<img src=x onerror=alert(1)> [Eval, p. 6]",
                ]
            }
        }

        with patch("views.notes.st", mock_st):
            notes_view.render_notes_view("fake_token", "test_doc")

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        joined_output = "\n".join(markdown_calls)

        self.assertIn("Q1: What is Q?", joined_output)
        self.assertIn("*Answer:* Query vector", joined_output)
        self.assertIn("Q2: What is K?", joined_output)
        self.assertIn("*Answer:* Key vector fallback", joined_output)
        self.assertIn("Q3: How is BLEU evaluated?", joined_output)
        self.assertIn("Q4:", joined_output)
        self.assertNotIn("<img", joined_output)
        self.assertIn("&lt;img", joined_output)
        self.assertIn("citation-chip", joined_output)

    def test_outline_view_dict_and_string_slides(self):
        from unittest.mock import MagicMock, patch
        import views.outline as outline_view

        mock_st = MagicMock()
        c1, c2, c3 = MagicMock(), MagicMock(), MagicMock()
        c1.slider.return_value = 8
        c2.button.return_value = False
        mock_st.columns.return_value = (c1, c2, c3)
        mock_st.session_state = {
            "outline_test_doc_8": {
                "title": "Transformer Presentation",
                "slides": [
                    {
                        "slide_number": 1,
                        "title": "Model Architecture",
                        "bullets": ["Attention [Model, p. 3]"],
                        "source": "[Model, p. 3]",
                    },
                    {
                        "title": "Training Setup",
                        "bullets": "Warmup and Adam optimizer [Setup, p. 5]",
                    },
                    "Conclusion and Future Directions [Discussion, p. 8]",
                ],
            }
        }

        with patch("views.outline.st", mock_st):
            outline_view.render_outline_view("fake_token", "test_doc")

        markdown_calls = [call.args[0] for call in mock_st.markdown.call_args_list if call.args]
        joined_output = "\n".join(markdown_calls)

        self.assertIn("Slide 1: Model Architecture", joined_output)
        self.assertIn("Slide 2: Training Setup", joined_output)
        self.assertIn("Slide 3: Slide 3", joined_output)
        self.assertIn("Conclusion and Future Directions", joined_output)
        self.assertIn("citation-chip", joined_output)

    def test_citation_formatting_and_html_cleaning_after_normalization(self):
        dirty_input = "<b onclick='bad()'>Bold</b> [Security, pp. 1-2]"
        cleaned = render_citations_to_html(dirty_input)
        self.assertNotIn("<b onclick", cleaned)
        self.assertIn("&lt;b onclick=&#x27;bad()&#x27;&gt;Bold&lt;/b&gt;", cleaned)
        self.assertIn("class=\"citation-chip\"", cleaned)
        self.assertIn("Security, pp. 1-2", cleaned)

        lenticular_input = "Claim with lenticular brackets 【Methods, p. 4】"
        lenticular_cleaned = render_citations_to_html(lenticular_input)
        self.assertIn("class=\"citation-chip\"", lenticular_cleaned)
        self.assertIn("Methods, p. 4", lenticular_cleaned)


if __name__ == "__main__":
    unittest.main()
