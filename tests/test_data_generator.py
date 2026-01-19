"""
Test Data Generator for Rlm Comprehensive Testing.

Generates various test prompts including:
- Needle in the haystack (256k tokens)
- Multi-needle tests
- Long context reasoning tests
"""

import random
import string
from typing import Tuple, List, Dict, Any


def generate_lorem_paragraph(min_words: int = 50, max_words: int = 150) -> str:
    """Generate a realistic-looking filler paragraph."""
    lorem_words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip",
        "ex", "ea", "commodo", "consequat", "duis", "aute", "irure", "in",
        "reprehenderit", "voluptate", "velit", "esse", "cillum", "fugiat", "nulla",
        "pariatur", "excepteur", "sint", "occaecat", "cupidatat", "non", "proident",
        "sunt", "culpa", "qui", "officia", "deserunt", "mollit", "anim", "id",
        "est", "laborum", "the", "and", "of", "to", "a", "in", "that", "is",
        "was", "for", "on", "are", "with", "as", "at", "be", "this", "have",
        "from", "or", "had", "by", "not", "but", "what", "all", "were", "when",
        "we", "there", "can", "an", "your", "which", "their", "said", "if", "will",
        "each", "about", "how", "up", "out", "them", "then", "she", "many", "some",
        "so", "these", "would", "other", "into", "has", "more", "her", "two", "like",
        "him", "see", "time", "could", "no", "make", "than", "first", "been", "its",
        "who", "now", "people", "my", "made", "over", "did", "down", "only", "way",
        "find", "use", "may", "water", "long", "little", "very", "after", "words",
        "called", "just", "where", "most", "know", "get", "through", "back", "much",
        "before", "go", "good", "new", "write", "our", "used", "me", "man", "too",
        "any", "day", "same", "right", "look", "think", "also", "around", "another",
        "came", "come", "work", "three", "word", "must", "because", "does", "part"
    ]

    num_words = random.randint(min_words, max_words)
    words = [random.choice(lorem_words) for _ in range(num_words)]

    # Capitalize first word and add some sentence structure
    words[0] = words[0].capitalize()
    sentences = []
    current_sentence = []

    for i, word in enumerate(words):
        current_sentence.append(word)
        # Random sentence length between 8-20 words
        if len(current_sentence) >= random.randint(8, 20) or i == len(words) - 1:
            sentence = " ".join(current_sentence)
            if current_sentence:
                sentence = sentence[0].upper() + sentence[1:]
            sentences.append(sentence + ".")
            current_sentence = []

    return " ".join(sentences)


def generate_document_section(section_num: int, topic: str = None) -> str:
    """Generate a document section with header and content."""
    topics = [
        "Financial Analysis", "Market Research", "Technical Documentation",
        "Project Report", "Research Findings", "Status Update", "Meeting Notes",
        "Policy Guidelines", "Process Description", "System Architecture",
        "User Requirements", "Testing Protocols", "Performance Metrics",
        "Risk Assessment", "Strategic Planning", "Operational Review",
        "Quality Assurance", "Compliance Report", "Budget Overview",
        "Resource Allocation", "Timeline Analysis", "Stakeholder Feedback"
    ]

    topic = topic or random.choice(topics)

    header = f"\n## Section {section_num}: {topic}\n\n"
    content = "\n\n".join([generate_lorem_paragraph() for _ in range(random.randint(3, 6))])

    return header + content


def generate_needle_haystack_prompt(
    target_tokens: int = 256_000,
    needle: str = None,
    needle_position: str = "middle",  # "start", "middle", "end", "random"
    question: str = None
) -> Tuple[str, str, str]:
    """
    Generate a needle-in-the-haystack test prompt.

    Args:
        target_tokens: Approximate number of tokens (chars/4)
        needle: The secret information to hide. Defaults to a magic code.
        needle_position: Where to place the needle
        question: Question to ask about the needle

    Returns:
        Tuple of (context, needle, question)
    """
    # Default needle - a distinctive piece of information
    if needle is None:
        needle = (
            "CRITICAL ALERT: The secret authorization code for Project Phoenix is "
            "'ALPHA-7749-OMEGA'. This code must be used for all level-5 clearance "
            "operations. Remember: ALPHA-7749-OMEGA is the key."
        )

    if question is None:
        question = "What is the secret authorization code for Project Phoenix?"

    # Estimate ~4 characters per token
    target_chars = target_tokens * 4
    needle_chars = len(needle)
    haystack_chars = target_chars - needle_chars - 1000  # Buffer for formatting

    # Generate sections
    sections = []
    chars_generated = 0
    section_num = 1

    while chars_generated < haystack_chars:
        section = generate_document_section(section_num)
        sections.append(section)
        chars_generated += len(section)
        section_num += 1

    # Insert needle at specified position
    total_sections = len(sections)

    if needle_position == "start":
        insert_pos = max(1, total_sections // 10)  # ~10% from start
    elif needle_position == "end":
        insert_pos = total_sections - max(1, total_sections // 10)  # ~10% from end
    elif needle_position == "random":
        insert_pos = random.randint(1, total_sections - 1)
    else:  # middle (default)
        insert_pos = total_sections // 2

    # Format needle as a special section
    needle_section = f"""
## Section {insert_pos}: Classified Information

{needle}

Please note that the above information is highly confidential and should be
memorized by authorized personnel only. Do not share this information.

"""

    # Insert needle
    sections.insert(insert_pos, needle_section)

    # Create final context
    header = """# Comprehensive Enterprise Documentation Package

This document contains extensive internal documentation, reports, and analysis
from various departments. All information is classified and for internal use only.

---

"""

    context = header + "\n".join(sections)

    return context, needle, question


def generate_multi_needle_prompt(
    target_tokens: int = 256_000,
    num_needles: int = 5
) -> Tuple[str, List[Dict[str, str]], str]:
    """
    Generate a test with multiple needles hidden throughout.

    Returns:
        Tuple of (context, needles_info, question)
    """
    needles_info = [
        {"key": "Project Alpha budget", "value": "$2.7 million", "code": "BUDGET-ALPHA-27M"},
        {"key": "Launch date for Beta", "value": "March 15th, 2025", "code": "DATE-BETA-0315"},
        {"key": "Lead engineer name", "value": "Dr. Sarah Chen", "code": "ENG-SARAH-CHEN"},
        {"key": "Server cluster count", "value": "47 nodes", "code": "CLUSTER-47-NODES"},
        {"key": "Security clearance level", "value": "Level 7 Omega", "code": "CLEAR-LVL7-OMEGA"},
    ][:num_needles]

    target_chars = target_tokens * 4
    chars_per_section = target_chars // (num_needles + 10)

    sections = []
    needle_positions = sorted(random.sample(range(2, num_needles + 10), num_needles))

    needle_idx = 0
    for section_num in range(1, num_needles + 12):
        if needle_idx < len(needle_positions) and section_num == needle_positions[needle_idx]:
            # Insert a needle with very clear, searchable format
            needle = needles_info[needle_idx]
            section = f"""
## Section {section_num}: Confidential Data Point

*** CLASSIFIED DATA POINT ***
Key: {needle['key']}
Value: {needle['value']}
Reference Code: {needle['code']}
*** END CLASSIFIED DATA ***

This information is restricted to authorized personnel with appropriate clearance.

"""
            sections.append(section)
            needle_idx += 1
        else:
            # Generate filler
            section = generate_document_section(section_num)
            # Trim to target size
            if len(section) > chars_per_section:
                section = section[:chars_per_section]
            sections.append(section)

    header = """# Multi-Department Classified Brief

This compilation contains sensitive information from multiple departments.
Each classified data point is marked between *** CLASSIFIED DATA POINT ***
and *** END CLASSIFIED DATA *** markers, containing Key, Value, and Reference Code.

---

"""

    context = header + "\n".join(sections)

    question = (
        "List ALL the classified data points mentioned in the document. "
        "For each one, provide the key, value, and reference code. "
        "Look for sections marked with *** CLASSIFIED DATA POINT ***."
    )

    return context, needles_info, question


def generate_reasoning_test_prompt(
    target_tokens: int = 50_000,
    complexity: str = "medium"  # "simple", "medium", "complex"
) -> Tuple[str, str, str]:
    """
    Generate a long context that requires reasoning to answer.

    Returns:
        Tuple of (context, expected_answer, question)
    """
    # Generate a story with interconnected facts - using clear FACT markers
    characters = [
        {"name": "Alice", "role": "CEO", "department": "Executive"},
        {"name": "Bob", "role": "CTO", "department": "Technology"},
        {"name": "Carol", "role": "CFO", "department": "Finance"},
        {"name": "David", "role": "COO", "department": "Operations"},
        {"name": "Eve", "role": "CMO", "department": "Marketing"},
    ]

    # Create relationship facts with explicit markers
    facts = [
        "[ORGANIZATIONAL FACT] Alice (CEO) reports to the board of directors and oversees all C-suite executives.",
        "[ORGANIZATIONAL FACT] Bob (CTO) manages the entire engineering team of 150 people.",
        "[ORGANIZATIONAL FACT] Carol (CFO) handles a budget of $50 million annually.",
        "[ORGANIZATIONAL FACT] David (COO) coordinates with all department heads for operational efficiency.",
        "[ORGANIZATIONAL FACT] Eve (CMO) leads a marketing team of 45 people with a budget of $12 million.",
        "[KEY COLLABORATION] Bob (CTO) and Carol (CFO) work together on technology investment decisions.",
        "[ORGANIZATIONAL FACT] Alice meets with David every Monday at 9 AM.",
        "[ORGANIZATIONAL FACT] Eve reports marketing metrics to Alice every Friday.",
        "[ORGANIZATIONAL FACT] Carol approves all budgets over $100,000.",
        "[ORGANIZATIONAL FACT] David implemented a new process that reduced costs by 15%.",
    ]

    if complexity == "complex":
        # Add more intricate relationships
        facts.extend([
            "[ORGANIZATIONAL FACT] Alice was previously the CTO before becoming CEO.",
            "[ORGANIZATIONAL FACT] Bob joined the company 5 years after David.",
            "[KEY AUTHORITY] Carol (CFO) has veto power over any expenditure exceeding $500,000.",
            "[ORGANIZATIONAL FACT] Eve's marketing campaign increased revenue by 23% last quarter.",
            "[ORGANIZATIONAL FACT] David's operations team includes a subset of Bob's engineers.",
        ])
        question = (
            "Based on the document, who has the authority to approve a $750,000 "
            "technology investment, and who else must be involved in that decision?"
        )
        expected_answer = (
            "Carol (CFO) has veto power over expenditures exceeding $500,000, "
            "so she must approve it. Bob (CTO) and Carol work together on technology "
            "investment decisions. Alice (CEO) oversees all C-suite executives and "
            "would have final authority."
        )
    elif complexity == "simple":
        question = "Who is the CTO and what is their primary responsibility?"
        expected_answer = "Bob is the CTO and manages the entire engineering team of 150 people."
    else:  # medium
        question = (
            "Which two executives collaborate on technology investment decisions, "
            "and what are their respective roles? Look for [KEY COLLABORATION] markers."
        )
        expected_answer = (
            "Bob (CTO) and Carol (CFO) work together on technology investment decisions. "
            "Bob manages the engineering team and Carol handles the $50 million annual budget."
        )

    # Generate context with facts embedded
    target_chars = target_tokens * 4
    chars_per_section = target_chars // 20

    sections = []
    fact_positions = sorted(random.sample(range(1, 19), len(facts)))

    fact_idx = 0
    for section_num in range(1, 21):
        if fact_idx < len(fact_positions) and section_num == fact_positions[fact_idx]:
            # Embed a fact with clear formatting
            fact = facts[fact_idx]
            section = f"""
## Section {section_num}: Organizational Update

{generate_lorem_paragraph(30, 50)}

--- BEGIN KEY INFORMATION ---
{fact}
--- END KEY INFORMATION ---

{generate_lorem_paragraph(30, 50)}

"""
            fact_idx += 1
        else:
            section = generate_document_section(section_num)

        if len(section) > chars_per_section:
            section = section[:chars_per_section]
        sections.append(section)

    header = """# Company Organizational Documentation

This document provides comprehensive information about the company's
organizational structure, key personnel, and operational procedures.

Key information is marked between --- BEGIN KEY INFORMATION --- and
--- END KEY INFORMATION --- markers.

---

"""

    context = header + "\n".join(sections)

    return context, expected_answer, question


def _create_technical_report(
    doc_num: int,
    report_type: str,
    key_points: List[str],
    target_chars: int
) -> str:
    """
    Create a single technical report with key points and filler content.

    Args:
        doc_num: Document number
        report_type: Type of report (e.g., "Q4 Performance Review")
        key_points: List of key point strings to include
        target_chars: Target character count for the document

    Returns:
        Formatted document string
    """
    sections = []

    # Document header
    header = f"""[DOCUMENT START - Document {doc_num}: {report_type}]

# {report_type}

## Executive Summary

"""
    sections.append(header)

    # Add first key point in executive summary
    if len(key_points) > 0:
        sections.append(f"[KEY POINT] {key_points[0]}\n\n")
        sections.append(generate_lorem_paragraph(40, 80) + "\n\n")

    # Add multiple sections with key points distributed
    section_count = random.randint(3, 5)
    key_point_idx = 1

    section_titles = [
        "Performance Metrics",
        "Technical Analysis",
        "Key Findings",
        "Recommendations",
        "Data Overview",
        "System Performance",
        "Implementation Status",
        "Results and Outcomes",
        "Risk Assessment",
        "Future Outlook"
    ]

    for i in range(section_count):
        section_title = random.choice(section_titles)
        sections.append(f"## {section_title}\n\n")

        # Add 2-4 paragraphs per section
        for _ in range(random.randint(2, 4)):
            sections.append(generate_lorem_paragraph(50, 100) + "\n\n")

            # Occasionally insert a key point
            if key_point_idx < len(key_points) and random.random() < 0.5:
                sections.append(f"[KEY POINT] {key_points[key_point_idx]}\n\n")
                sections.append(generate_lorem_paragraph(30, 60) + "\n\n")
                key_point_idx += 1

    # Add remaining key points in conclusion
    if key_point_idx < len(key_points):
        sections.append("## Conclusion\n\n")
        sections.append(generate_lorem_paragraph(40, 60) + "\n\n")
        while key_point_idx < len(key_points):
            sections.append(f"[KEY POINT] {key_points[key_point_idx]}\n\n")
            sections.append(generate_lorem_paragraph(20, 40) + "\n\n")
            key_point_idx += 1

    # Document footer
    sections.append(f"\n[DOCUMENT END]\n\n")

    # Combine all sections
    document = "".join(sections)

    # Pad with filler if needed to reach target size
    current_size = len(document)
    if current_size < target_chars:
        sections_to_add = (target_chars - current_size) // 800  # Approximate chars per paragraph
        filler_section = "\n## Additional Details\n\n"
        for _ in range(sections_to_add):
            filler_section += generate_lorem_paragraph(80, 120) + "\n\n"
        # Insert filler before the end marker
        document = document.replace("[DOCUMENT END]", filler_section + "[DOCUMENT END]")

    return document


def generate_summarization_prompt(
    target_tokens: int = 100_000,
    num_documents: int = None
) -> Tuple[str, List[str], str]:
    """
    Generate a long text summarization test with multiple technical reports.

    This creates 4-6 realistic technical reports with clear structure and key points
    that need to be summarized. Each document includes:
    - Clear document boundaries with [DOCUMENT START] and [DOCUMENT END] markers
    - Key points marked with [KEY POINT] tags
    - Realistic filler content
    - Technical details (metrics, percentages, numbers)

    Args:
        target_tokens: Approximate number of tokens (chars/4), default 100k
        num_documents: Number of documents to generate (default: 4-6 random)

    Returns:
        Tuple of (context, expected_key_points, task)
        - context: All technical reports concatenated
        - expected_key_points: List of key information strings
        - task: The summarization task description
    """
    if num_documents is None:
        num_documents = random.randint(4, 6)

    # Define technical reports with key points
    report_templates = [
        {
            "type": "Q4 Financial Performance Review",
            "key_points": [
                "Q4 revenue increased by 23% to $4.2M compared to Q3",
                "Customer churn reduced from 8% to 3% through improved support",
                "Operating margin improved to 18.5% from 15.2% previous quarter"
            ]
        },
        {
            "type": "Engineering System Update",
            "key_points": [
                "New API deployed with 99.9% uptime across all regions",
                "Migration to microservices architecture completed ahead of schedule",
                "Database query performance improved by 40% through indexing optimization"
            ]
        },
        {
            "type": "Product Development Roadmap",
            "key_points": [
                "Feature X launching in Q1 2026 with beta testing complete",
                "Beta testing showed 85% user satisfaction rate",
                "Mobile app development on track for March 2026 release"
            ]
        },
        {
            "type": "Customer Analytics Report",
            "key_points": [
                "Active user base grew to 125,000 users (up 15% month-over-month)",
                "Average session duration increased to 18 minutes from 12 minutes",
                "Net Promoter Score improved to 67 from 54"
            ]
        },
        {
            "type": "Infrastructure Security Audit",
            "key_points": [
                "Zero critical vulnerabilities found in latest security audit",
                "SSL certificate renewal automated for all 47 services",
                "Multi-factor authentication adoption reached 92% of users"
            ]
        },
        {
            "type": "Marketing Campaign Analysis",
            "key_points": [
                "Email campaign achieved 28% open rate and 6.5% click-through rate",
                "Social media engagement up 156% following new content strategy",
                "Cost per acquisition reduced to $45 from $78"
            ]
        },
    ]

    # Select documents to include
    selected_reports = random.sample(report_templates, min(num_documents, len(report_templates)))

    # Calculate target size per document
    target_chars = target_tokens * 4
    chars_per_doc = target_chars // num_documents

    # Generate all documents
    documents = []
    all_key_points = []

    for idx, report in enumerate(selected_reports, 1):
        doc = _create_technical_report(
            doc_num=idx,
            report_type=report["type"],
            key_points=report["key_points"],
            target_chars=chars_per_doc
        )
        documents.append(doc)
        all_key_points.extend(report["key_points"])

    # Create header
    header = """# Technical Reports Collection

This collection contains multiple technical reports from various departments.
Each report is marked with [DOCUMENT START] and [DOCUMENT END] boundaries.
Key findings and metrics are marked with [KEY POINT] tags.

Please read all reports carefully to provide a comprehensive summary.

---

"""

    # Combine all documents
    context = header + "\n".join(documents)

    # Create task
    task = (
        "Provide a comprehensive technical summary of all reports, highlighting the "
        "key findings, metrics, and recommendations. Focus on the most important "
        "information marked with [KEY POINT] tags."
    )

    return context, all_key_points, task


def generate_edge_case_prompts() -> List[Dict[str, Any]]:
    """
    Generate a list of edge case test prompts.

    Returns:
        List of test cases with context, question, and expected behavior
    """
    return [
        {
            "name": "empty_needle",
            "context": generate_lorem_paragraph(500, 1000),
            "question": "What is the secret code mentioned in the document?",
            "expected": "no_answer",  # Should say no secret code exists
            "description": "Tests handling of questions about non-existent information"
        },
        {
            "name": "repeated_info",
            "context": (
                "--- DOCUMENT START ---\n" +
                "[DECOY] The password is ABC123.\n" * 50 +
                generate_lorem_paragraph(200, 300) + "\n" +
                "[DECOY] The password is ABC123.\n" * 50 +
                "\n*** ACTUAL PASSWORD ***\nThe real password is: XYZ789\n*** END ACTUAL PASSWORD ***\n" +
                generate_lorem_paragraph(200, 300) +
                "\n--- DOCUMENT END ---"
            ),
            "question": "What is the real password? Look for *** ACTUAL PASSWORD *** marker, not the [DECOY] entries.",
            "expected": "XYZ789",
            "description": "Tests distinguishing repeated vs unique information"
        },
        {
            "name": "contradictory_info",
            "context": (
                "--- DOCUMENT START ---\n\n" +
                "[ORIGINAL] Section 1: The meeting is scheduled for Monday.\n\n" +
                generate_lorem_paragraph(200, 300) +
                "\n\n[CORRECTION] Section 2: CORRECTION - The meeting has been moved to Wednesday.\n\n" +
                generate_lorem_paragraph(200, 300) +
                "\n--- DOCUMENT END ---"
            ),
            "question": "When is the meeting scheduled? Note: [CORRECTION] entries override [ORIGINAL] entries.",
            "expected": "Wednesday",
            "description": "Tests handling contradictory information (should use latest/correction)"
        },
        {
            "name": "numeric_precision",
            "context": (
                "--- FINANCIAL REPORT ---\n" +
                "Q1 Revenue: $1,234,567.89\n" +
                "Q2 Revenue: $2,345,678.90\n" +
                "Q3 Revenue: $3,456,789.01\n" +
                "Q4 Revenue: $4,567,890.12\n" +
                "--- END FINANCIAL REPORT ---\n\n" +
                generate_lorem_paragraph(500, 800)
            ),
            "question": "What is the total annual revenue? Sum Q1+Q2+Q3+Q4.",
            "expected": "$11,604,925.92",
            "description": "Tests numeric computation with precision"
        },
    ]


# Convenience functions for generating test files
def save_test_prompt(filepath: str, context: str, question: str, metadata: Dict = None):
    """Save a test prompt to a file."""
    import json

    data = {
        "context": context,
        "question": question,
        "metadata": metadata or {},
        "context_length_chars": len(context),
        "estimated_tokens": len(context) // 4
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    # Generate sample test data when run directly
    print("Generating 256k needle-in-the-haystack test...")
    context, needle, question = generate_needle_haystack_prompt(target_tokens=256_000)
    print(f"Generated context: {len(context):,} characters (~{len(context)//4:,} tokens)")
    print(f"Needle: {needle[:100]}...")
    print(f"Question: {question}")
