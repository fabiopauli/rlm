"""
Advanced helper functions for RLM operations.

Provides utilities for text processing, chunking, searching, and aggregation.
"""

import re
from typing import List, Dict, Any, Callable, Optional, Tuple
import hashlib


class TextProcessor:
    """Advanced text processing utilities."""

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 2000,
        overlap: int = 200,
        preserve_paragraphs: bool = False
    ) -> List[str]:
        """
        Chunk text with optional overlap and paragraph preservation.

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            preserve_paragraphs: If True, try to break at paragraph boundaries

        Returns:
            List of text chunks
        """
        if not preserve_paragraphs:
            # Simple character-based chunking
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks

        # Paragraph-preserving chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If single paragraph is larger than chunk_size, split it
                if len(para) > chunk_size:
                    para_chunks = TextProcessor.chunk_text(para, chunk_size, overlap, False)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def chunk_by_tokens(
        text: str,
        max_tokens: int = 1000,
        overlap_tokens: int = 100,
        encoding_name: str = "cl100k_base"
    ) -> List[str]:
        """
        Chunk text by token count (more accurate for LLM context limits).

        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Tokens to overlap between chunks
            encoding_name: Tiktoken encoding name

        Returns:
            List of text chunks

        Note:
            Requires 'tiktoken' package. Falls back to character-based if not available.
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding(encoding_name)

            # Encode text
            tokens = encoding.encode(text)

            chunks = []
            start = 0

            while start < len(tokens):
                end = start + max_tokens
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                start = end - overlap_tokens

            return chunks

        except ImportError:
            # Fallback to character-based chunking
            # Rough approximation: 1 token ≈ 4 characters
            char_size = max_tokens * 4
            char_overlap = overlap_tokens * 4
            return TextProcessor.chunk_text(text, char_size, char_overlap)

    @staticmethod
    def smart_truncate(text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to max length, trying to break at word boundaries.

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Try to break at last space before max_length
        truncated = text[:max_length - len(suffix)]
        last_space = truncated.rfind(' ')

        if last_space > 0:
            truncated = truncated[:last_space]

        return truncated + suffix


class SearchHelper:
    """Advanced search and filtering utilities."""

    @staticmethod
    def regex_search(
        pattern: str,
        text: str,
        max_matches: Optional[int] = None,
        return_positions: bool = False,
        flags: int = 0
    ) -> List[Any]:
        """
        Perform regex search with optional limits.

        Args:
            pattern: Regex pattern
            text: Text to search
            max_matches: Maximum number of matches to return
            return_positions: If True, return dicts with 'match', 'start', 'end' keys
            flags: Regex flags (e.g., re.MULTILINE, re.DOTALL, re.IGNORECASE)

        Returns:
            List of matches (strings) or match info dicts

        Note:
            For simple tag matching, prefer find_tags() for better reliability and performance.
        """
        matches = []

        for i, match in enumerate(re.finditer(pattern, text, flags)):
            if max_matches and i >= max_matches:
                break

            if return_positions:
                matches.append({
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
            else:
                matches.append(match.group())

        return matches

    @staticmethod
    def find_sections(
        text: str,
        section_pattern: str = r'^#+\s+(.+)$',
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find sections in structured text (e.g., markdown headers).

        Args:
            text: Text to search
            section_pattern: Regex pattern for section headers
            include_content: If True, include content between sections

        Returns:
            List of section dictionaries
        """
        sections = []
        lines = text.split('\n')

        current_section = None
        current_content = []

        for i, line in enumerate(lines):
            match = re.match(section_pattern, line, re.MULTILINE)

            if match:
                # Save previous section
                if current_section and include_content:
                    current_section['content'] = '\n'.join(current_content)
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'title': match.group(1),
                    'line_number': i + 1,
                    'header': line
                }
                current_content = []

            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            if include_content:
                current_section['content'] = '\n'.join(current_content)
            sections.append(current_section)

        return sections

    @staticmethod
    def keyword_filter(
        text: str,
        keywords: List[str],
        context_chars: int = 200,
        case_sensitive: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Filter text by keywords and return matching snippets with context.

        Args:
            text: Text to search
            keywords: List of keywords to search for
            context_chars: Characters of context around matches
            case_sensitive: Whether search is case-sensitive

        Returns:
            List of (snippet, position) tuples
        """
        snippets = []
        text_search = text if case_sensitive else text.lower()

        for keyword in keywords:
            kw_search = keyword if case_sensitive else keyword.lower()
            start = 0

            while True:
                pos = text_search.find(kw_search, start)
                if pos == -1:
                    break

                # Extract context
                snippet_start = max(0, pos - context_chars)
                snippet_end = min(len(text), pos + len(keyword) + context_chars)
                snippet = text[snippet_start:snippet_end]

                snippets.append((snippet, pos))
                start = pos + 1

        # Sort by position and remove duplicates
        snippets = sorted(set(snippets), key=lambda x: x[1])

        return snippets

    @staticmethod
    def find_tags(
        text: str,
        tag_pattern: str,
        extract_content: bool = True,
        content_delimiter: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all occurrences of a tag pattern using fast string search.

        This is more reliable than regex for simple tag matching and works
        correctly even when multiple tags appear close together or on the same line.

        Args:
            text: Text to search
            tag_pattern: Tag pattern to find (e.g., "[KEY POINT]")
            extract_content: If True, extract content after the tag
            content_delimiter: Stop extracting content at this delimiter
                             (if None, extracts until next tag or end of text)
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with 'tag', 'content', 'start', 'end' keys

        Example:
            >>> text = "before [KEY POINT] first info [KEY POINT] second info end"
            >>> tags = find_tags(text, "[KEY POINT]", extract_content=True)
            >>> len(tags)
            2
            >>> "first info" in tags[0]['content']
            True
        """
        results = []
        start_pos = 0

        while True:
            # Find next occurrence
            pos = text.find(tag_pattern, start_pos)
            if pos == -1:
                break

            tag_end = pos + len(tag_pattern)
            content = ""
            content_end = tag_end

            if extract_content:
                # Extract content after tag
                content_start = tag_end

                # Find where content ends
                if content_delimiter:
                    # Stop at delimiter
                    delim_pos = text.find(content_delimiter, content_start)
                    if delim_pos != -1:
                        content_end = delim_pos
                    else:
                        # Find next tag occurrence or end of text
                        next_tag_pos = text.find(tag_pattern, tag_end)
                        content_end = next_tag_pos if next_tag_pos != -1 else len(text)
                else:
                    # Stop at next tag or end of text
                    next_tag_pos = text.find(tag_pattern, tag_end)
                    content_end = next_tag_pos if next_tag_pos != -1 else len(text)

                content = text[content_start:content_end].strip()

            results.append({
                'tag': tag_pattern,
                'content': content,
                'start': pos,
                'end': content_end if extract_content else tag_end
            })

            # Check if we've hit max results
            if max_results and len(results) >= max_results:
                break

            # Move past this tag for next search
            start_pos = tag_end

        return results

    @staticmethod
    def extract_between_markers(
        text: str,
        start_marker: str,
        end_marker: str,
        include_markers: bool = False,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract content between paired start and end markers.

        Args:
            text: Text to search
            start_marker: Starting marker (e.g., "[DOCUMENT START]")
            end_marker: Ending marker (e.g., "[DOCUMENT END]")
            include_markers: If True, include markers in extracted content
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with 'content', 'start', 'end' keys

        Example:
            >>> text = "[DOC START]content1[DOC END] [DOC START]content2[DOC END]"
            >>> docs = extract_between_markers(text, "[DOC START]", "[DOC END]")
            >>> len(docs)
            2
            >>> docs[0]['content']
            'content1'
        """
        results = []
        start_pos = 0

        while True:
            # Find next start marker
            start_idx = text.find(start_marker, start_pos)
            if start_idx == -1:
                break

            # Find corresponding end marker
            end_search_start = start_idx + len(start_marker)
            end_idx = text.find(end_marker, end_search_start)
            if end_idx == -1:
                # No matching end marker found
                break

            # Extract content
            if include_markers:
                content_start = start_idx
                content_end = end_idx + len(end_marker)
            else:
                content_start = start_idx + len(start_marker)
                content_end = end_idx

            content = text[content_start:content_end]

            results.append({
                'content': content,
                'start': start_idx,
                'end': end_idx + len(end_marker)
            })

            # Check if we've hit max results
            if max_results and len(results) >= max_results:
                break

            # Move past this pair for next search
            start_pos = end_idx + len(end_marker)

        return results

    @staticmethod
    def count_occurrences(
        text: str,
        pattern: str,
        overlapping: bool = False
    ) -> int:
        """
        Count occurrences of a pattern using fast string search.

        Args:
            text: Text to search
            pattern: Pattern to count
            overlapping: If True, count overlapping occurrences

        Returns:
            Number of occurrences

        Example:
            >>> text = "[KEY POINT] info [KEY POINT] more"
            >>> count_occurrences(text, "[KEY POINT]")
            2
            >>> count_occurrences("aaa", "aa", overlapping=True)
            2
            >>> count_occurrences("aaa", "aa", overlapping=False)
            1
        """
        if not pattern:
            return 0

        count = 0
        start_pos = 0

        while True:
            pos = text.find(pattern, start_pos)
            if pos == -1:
                break

            count += 1

            # Move position for next search
            if overlapping:
                start_pos = pos + 1
            else:
                start_pos = pos + len(pattern)

        return count


class AggregationHelper:
    """Utilities for aggregating results from sub-calls."""

    @staticmethod
    def aggregate_results(
        results: List[Any],
        method: str = 'join',
        separator: str = '\n',
        filter_empty: bool = True
    ) -> Any:
        """
        Aggregate a list of results using various methods.

        Args:
            results: List of results to aggregate
            method: Aggregation method ('join', 'sum', 'count', 'list', 'dict')
            separator: Separator for 'join' method
            filter_empty: If True, filter out empty/None results

        Returns:
            Aggregated result
        """
        if filter_empty:
            results = [r for r in results if r]

        if method == 'join':
            return separator.join(str(r) for r in results)
        elif method == 'sum':
            return sum(results)
        elif method == 'count':
            return len(results)
        elif method == 'list':
            return results
        elif method == 'dict':
            # Assumes results are (key, value) tuples
            return dict(results)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    @staticmethod
    def count_frequencies(items: List[str]) -> Dict[str, int]:
        """
        Count frequency of items.

        Args:
            items: List of items

        Returns:
            Dictionary of item frequencies
        """
        frequencies = {}
        for item in items:
            frequencies[item] = frequencies.get(item, 0) + 1
        return frequencies

    @staticmethod
    def merge_dicts(dicts: List[Dict], merge_strategy: str = 'sum') -> Dict:
        """
        Merge multiple dictionaries.

        Args:
            dicts: List of dictionaries to merge
            merge_strategy: How to handle duplicate keys ('sum', 'last', 'first', 'list')

        Returns:
            Merged dictionary
        """
        if merge_strategy == 'last':
            result = {}
            for d in dicts:
                result.update(d)
            return result

        elif merge_strategy == 'first':
            result = {}
            for d in dicts:
                for k, v in d.items():
                    if k not in result:
                        result[k] = v
            return result

        elif merge_strategy == 'sum':
            result = {}
            for d in dicts:
                for k, v in d.items():
                    result[k] = result.get(k, 0) + v
            return result

        elif merge_strategy == 'list':
            result = {}
            for d in dicts:
                for k, v in d.items():
                    if k not in result:
                        result[k] = []
                    result[k].append(v)
            return result

        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")


class VerificationHelper:
    """Utilities for verification and validation of sub-call results."""

    @staticmethod
    def verify_answer(
        answer: str,
        verification_prompt: str,
        llm_query: Callable,
        threshold: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify an answer using a sub-call.

        Args:
            answer: The answer to verify
            verification_prompt: Prompt template for verification
            llm_query: Function to call LLM
            threshold: Optional threshold keyword (e.g., "yes", "correct")

        Returns:
            Tuple of (is_valid, explanation)
        """
        prompt = f"{verification_prompt}\n\nAnswer to verify: {answer}\n\nIs this correct? Respond with 'YES' or 'NO' and explain."
        response = llm_query(prompt)

        # Check for threshold keyword
        is_valid = False
        if threshold:
            is_valid = threshold.lower() in response.lower()
        else:
            is_valid = 'yes' in response.lower()

        return is_valid, response

    @staticmethod
    def consensus_check(
        question: str,
        llm_query: Callable,
        num_attempts: int = 3
    ) -> Tuple[str, float]:
        """
        Get consensus answer by querying multiple times.

        Args:
            question: Question to ask
            llm_query: Function to call LLM
            num_attempts: Number of attempts

        Returns:
            Tuple of (consensus_answer, confidence)
        """
        answers = []
        for _ in range(num_attempts):
            response = llm_query(question)
            answers.append(response.strip())

        # Find most common answer
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]

        confidence = count / num_attempts

        return most_common_answer, confidence


class RecursionHelper:
    """Utilities for recursive decomposition patterns."""

    @staticmethod
    def recursive_split(
        text: str,
        condition: Callable[[str], bool],
        split_fn: Callable[[str], List[str]],
        max_depth: int = 10,
        current_depth: int = 0
    ) -> List[str]:
        """
        Recursively split text until condition is met.

        Args:
            text: Text to split
            condition: Function to check if split is needed (returns True if small enough)
            split_fn: Function to split text
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth

        Returns:
            List of text chunks that satisfy condition
        """
        if current_depth >= max_depth or condition(text):
            return [text]

        # Split and recurse
        chunks = split_fn(text)
        result = []

        for chunk in chunks:
            result.extend(
                RecursionHelper.recursive_split(
                    chunk, condition, split_fn, max_depth, current_depth + 1
                )
            )

        return result

    @staticmethod
    def map_reduce(
        items: List[Any],
        map_fn: Callable,
        reduce_fn: Callable,
        parallel: bool = False
    ) -> Any:
        """
        Map-reduce pattern for processing items.

        Args:
            items: Items to process
            map_fn: Function to apply to each item
            reduce_fn: Function to reduce results
            parallel: If True, use parallel processing (requires concurrent.futures)

        Returns:
            Reduced result
        """
        if parallel:
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor() as executor:
                    mapped = list(executor.map(map_fn, items))
            except ImportError:
                mapped = [map_fn(item) for item in items]
        else:
            mapped = [map_fn(item) for item in items]

        return reduce_fn(mapped)
