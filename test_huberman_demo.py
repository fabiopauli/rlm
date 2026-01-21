"""
Demonstration of RLM processing on Huberman Lab podcast transcript
"""
import re

# Read the transcript
with open('huberman_transcript.txt', 'r') as f:
    transcript = f.read()

print("=" * 70)
print("RLM HUBERMAN TRANSCRIPT ANALYSIS - DEMONSTRATION")
print("=" * 70)
print()
print(f"Transcript length: {len(transcript):,} characters")
print(f"Word count: {len(transcript.split()):,} words")
print(f"Estimated tokens: ~{len(transcript) // 4:,}")
print()

# Show what RLM would analyze
print("TASK DEFINITION:")
print("-" * 70)
task = """Extract and summarize:
1) The main concepts about the nervous system discussed in this podcast
2) Any practical tools or protocols mentioned that listeners can apply  
3) Key scientific insights about neuroplasticity and learning"""
print(task)
print()

print("CONTEXT SAMPLE:")
print("-" * 70)
print(transcript[:500])
print("...\n")

print("=" * 70)
print("RLM PROCESSING STRUCTURE (How it works):")
print("=" * 70)
print()
print("1. INITIALIZATION")
print("   ✓ Load transcript into REPL as 'context' variable")
print("   ✓ Task requires multi-part extraction and summarization")
print("   ✓ Context fits in window but needs recursive processing")
print()
print("2. ROOT LLM STRATEGY GENERATION")
print("   The root LLM would generate Python code like:")
print("""
   # Find sections using text search
   sections = find_sections(context, r'^-\\s+(.+)$')
   
   # Process each section with sub-LLM calls
   concepts = []
   tools = []
   insights = []
   
   for section in sections:
       result = llm_query(f"Extract key concepts from: {section[:1000]}")
       concepts.append(result)
       
       tool_result = llm_query(f"Find practical protocols in: {section[:1000]}")
       tools.append(tool_result)
   
   # Aggregate and format
   final_summary = aggregate_results([concepts, tools, insights], method='join')
   FINAL(final_summary)
   """)
print()
print("3. SUB-CALL EXECUTION")
print("   ✓ Each llm_query() is a recursive LLM call on small chunks")
print("   ✓ Sub-calls focus on specific semantic tasks")
print("   ✓ Results are cached to avoid redundant calls")
print()
print("4. EXPECTED METRICS")
print("   - Sub-calls: 10-15 (one per major section)")
print("   - Total tokens: 20,000-30,000")
print("   - Cost estimate: $0.05-0.15 (with Grok)")
print("   - Iterations: 3-5 (until FINAL() called)")
print()

print("=" * 70)
print("QUICK TEXT ANALYSIS (What RLM would extract):")
print("=" * 70)
print()

# Find key concepts mentioned
key_terms = {
    'neurons': 0, 'synapse': 0, 'plasticity': 0, 'neuroplasticity': 0,
    'autonomic': 0, 'protocol': 0, 'practice': 0, 'learning': 0,
    'sleep': 0, 'focus': 0, 'attention': 0, 'nervous system': 0
}

for term in key_terms:
    pattern = r'\b' + term.replace(' ', r'\s+') + r'\w*\b'
    key_terms[term] = len(re.findall(pattern, transcript.lower()))

print("KEY TERMS FREQUENCY:")
for term, count in sorted(key_terms.items(), key=lambda x: x[1], reverse=True)[:10]:
    if count > 0:
        print(f"  • {term}: {count} mentions")
print()

# Find protocol/tool mentions
protocol_keywords = ['protocol', 'practice', 'tool', 'technique', 'exercise', 'breathing']
print("PRACTICAL TOOLS INDICATORS:")
for keyword in protocol_keywords:
    count = len(re.findall(r'\b' + keyword + r'\w*\b', transcript.lower()))
    if count > 0:
        print(f"  • {keyword}: {count} mentions")
print()

# Extract some sample sentences mentioning protocols
print("SAMPLE CONTENT (Protocol-related):")
sentences = re.split(r'[.!?]\s+', transcript)
protocol_sentences = [s for s in sentences if any(kw in s.lower() for kw in ['protocol', 'practice', 'can do'])][:3]
for i, sent in enumerate(protocol_sentences, 1):
    print(f"{i}. {sent[:150]}...")
print()

print("=" * 70)
print("RLM ADVANTAGES FOR THIS TASK:")
print("=" * 70)
print()
print("✓ Handles long transcript beyond single context window")
print("✓ Recursively processes each section for deep extraction")
print("✓ Separates concerns: structure detection (code) vs semantics (LLM)")
print("✓ Caches repeated sub-queries (e.g., similar concepts)")
print("✓ Provides detailed metrics on token usage and costs")
print("✓ Generates structured, categorized output")
print()
print("Full RLM run would provide comprehensive, organized summary of:")
print("  1. All major nervous system concepts explained")
print("  2. Complete list of actionable protocols with details")
print("  3. Key scientific insights with supporting evidence")
print()
