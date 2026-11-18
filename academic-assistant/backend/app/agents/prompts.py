"""Agent prompt templates for summarization, notes, and presentation outlines."""

SECTION_SUMMARY_PROMPT = """You are an academic research assistant summarizing a section of a research paper.
Synthesize the key points, findings, and methodology discussed in this section.
Use ONLY facts stated directly in the text below.
When citing facts, cite the section and page using format [Section, p. X] or [Section, pp. X-Y].
Rules:
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
- If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
- Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
- Do not add praise, conclusions or editorial claims the paper does not make.
- Output ONLY the synthesis without meta commentary about the excerpts, field names, or the output itself.

Section Name: {section_name}

Text:
{text}

Provide a concise, coherent summary of this section with citations.
"""

MAP_SUMMARY_PROMPT = """You are an academic research assistant.
Summarize the following excerpt from section '{section_name}'.
Maintain accurate citations in the format [Section, p. X] or [Section, pp. X-Y].
Rules:
- Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
- If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
- Do not add praise, conclusions or editorial claims the paper does not make.

Text:
{text}

Summary:
"""

REDUCE_SUMMARY_PROMPT = """You are an academic research assistant.
Combine the following partial summaries of section '{section_name}' into a unified, coherent section summary.
Preserve key facts and all citations in the format [Section, p. X] or [Section, pp. X-Y].
Rules:
- Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
- If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
- Do not add praise, conclusions or editorial claims the paper does not make.

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
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
- If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
- Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
- Do not add praise, conclusions or editorial claims the paper does not make.
- Output ONLY the executive summary without meta commentary, disclaimer sentences about excerpts, or field labels.

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
Rules:
- Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
- If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
- Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
- Do not add praise, conclusions or editorial claims the paper does not make.
- Do not invent information. If the excerpts do not contain enough information, state that clearly.
- Output ONLY the factual synthesis. Do NOT include meta commentary, disclaimers about excerpts, or field markers like "(Field: ...)".
"""

KEY_TERMS_PROMPT = """You are an academic research assistant.
Based on the following excerpts from a research paper, identify 4 to 8 key technical concepts or terminology introduced or relied upon.
Requirements:
- Define terms ONLY from the excerpts. If the excerpts do not define a term, give the term with the context in which the paper uses it, not outside knowledge.
- Do not add praise, conclusions or editorial claims the paper does not make.

Excerpts:
{excerpts}

Return a valid JSON object matching this schema:
{{
  "key_terms": [
    {{"term": "<term name>", "definition": "<concise definition grounded strictly in excerpts>"}}
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
You are provided with the first-page publication context, document summary, and structured notes.

First-Page Context:
{first_page_context}

Summary:
{summary}

Notes:
{notes}

Requirements:
- Generate a presentation outline containing EXACTLY {num_slides} slides.
- Slide numbers MUST be consecutively numbered from 1 to {num_slides} (slide_number: 1, 2, ..., {num_slides}).
- Slide 1 is the Title Slide:
  * The title slide may use ONLY the title and authors that appear in the First-Page Context above. If authors are not found, show only the title.
  * NEVER output a venue, year, or conference unless it explicitly appears in the First-Page Context.
- The Limitations slide (if included):
  * MUST be built ONLY from the limitations note. Its source_fields must be ["limitations"].
- Every slide must draw from valid source_fields: ["problem", "method", "dataset_and_setup", "key_results", "limitations"].
- Rules for slide bullet points:
  * Do not state as fact anything the excerpts present as a hypothesis or possibility (may, might, we hypothesize, we expect, we plan to). Report it as the authors state it.
  * If the excerpts give two different numbers for the same result, report both as stated. Do not reconcile or explain the difference.
  * Do not add detail the excerpts do not state, for example 'per model', 'each', 'all' or 'on average', unless the excerpts say so.
  * Do not add praise, conclusions or editorial claims the paper does not make.
- Output ONLY the valid JSON object without meta commentary.

Return a valid JSON object matching this schema:
{{
  "title": "<Presentation Title from First-Page Context or Summary>",
  "slides": [
    {{
      "slide_number": 1,
      "title": "<Slide Title>",
      "bullets": ["<Bullet 1>", "<Bullet 2>", "<Bullet 3>"],
      "source_fields": ["problem"]
    }}
  ]
}}
"""
