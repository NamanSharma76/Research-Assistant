"""Agent prompt templates for summarization, notes, and presentation outlines."""

SECTION_SUMMARY_PROMPT = """You are an academic research assistant summarizing a section of a research paper.
Synthesize the key points, findings, and methodology discussed in this section.
Use ONLY facts stated directly in the text below.
When citing facts, cite the section and page using format [Section, p. X] or [Section, pp. X-Y].
Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we plan to). Report it as the authors state it. If the paper gives two different numbers for the same result, mention both. Do not add praise or editorial claims.

Section Name: {section_name}

Text:
{text}

Provide a concise, coherent summary of this section with citations.
"""

MAP_SUMMARY_PROMPT = """You are an academic research assistant.
Summarize the following excerpt from section '{section_name}'.
Maintain accurate citations in the format [Section, p. X] or [Section, pp. X-Y].

Text:
{text}

Summary:
"""

REDUCE_SUMMARY_PROMPT = """You are an academic research assistant.
Combine the following partial summaries of section '{section_name}' into a unified, coherent section summary.
Preserve key facts and all citations in the format [Section, p. X] or [Section, pp. X-Y].

Partial Summaries:
{text}

Unified Summary:
"""

DOCUMENT_OVERALL_SUMMARY_PROMPT = """You are an academic research assistant.
Below are individual section summaries from a research paper.
Synthesize an overall executive summary (3-5 paragraphs) covering:
1. The core problem and motivations
2. The primary method and architectural innovations
3. Key experimental findings and comparative results
4. Significance and implications

Requirements:
- Ground all facts strictly in the section summaries provided below.
- You MUST include accurate citations in the format [Section, p. X] or [Section, pp. X-Y] drawn directly from the section summaries.
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we plan to). Report it as the authors state it. If the paper gives two different numbers for the same result, mention both. Do not add praise or editorial claims.

Section Summaries:
{section_summaries}

Produce a well-structured executive summary with citations.
"""

NOTE_FIELD_PROMPT = """You are an academic research assistant preparing structured study notes from a research paper.
Field to generate: {field_name}
Target description: {field_description}

Relevant Excerpts:
{excerpts}

Instructions:
Extract and synthesize the relevant details strictly grounded in the excerpts above.
Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we plan to). Report it as the authors state it. If the paper gives two different numbers for the same result, mention both. Do not add praise or editorial claims.
Do not invent information. If the excerpts do not contain enough information, state that clearly.
Output plain text synthesizing the answer.
"""

KEY_TERMS_PROMPT = """You are an academic research assistant.
Based on the following excerpts from a research paper, identify 4 to 8 key technical concepts or terminology introduced or relied upon.
Provide a concise, rigorous definition grounded strictly in the text for each.

Excerpts:
{excerpts}

Return a valid JSON object matching this schema:
{{
  "key_terms": [
    {{"term": "<term name>", "definition": "<concise definition>"}}
  ]
}}
"""

REVISION_QUESTIONS_PROMPT = """You are an academic research assistant.
Based on the following excerpts from a research paper, generate 4 to 6 thoughtful revision questions that test a student's deep conceptual understanding of the paper's methods, experimental setup, or results.

Excerpts:
{excerpts}

Return a valid JSON object matching this schema:
{{
  "revision_questions": [
    "<question 1>",
    "<question 2>"
  ]
}}
"""

OUTLINE_PROMPT = """You are an academic research assistant designing a conference presentation outline based on a research paper.
You are provided with the document summary and structured notes.

Summary:
{summary}

Notes:
{notes}

Requirements:
- Generate a presentation outline containing EXACTLY {num_slides} slides.
- Slide numbers MUST be consecutively numbered from 1 to {num_slides} (slide_number: 1, 2, ..., {num_slides}).
- Each slide must have a clear title, 3 to 5 concise bullet points, and a list of source_fields from the notes it draws on: ["problem", "method", "dataset_and_setup", "key_results", "limitations"].
- Cover Title/Introduction, Motivation/Problem, Method/Architecture, Experiments/Results, and Conclusion/Takeaways logically distributed across the {num_slides} slides.
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we plan to). Report it as the authors state it. If the paper gives two different numbers for the same result, mention both. Do not add praise or editorial claims.

Return a valid JSON object matching this schema:
{{
  "title": "<Presentation Title>",
  "slides": [
    {{
      "slide_number": 1,
      "title": "<Slide Title>",
      "bullets": ["<Bullet 1>", "<Bullet 2>", "<Bullet 3>"],
      "source_fields": ["method"]
    }}
  ]
}}
"""
