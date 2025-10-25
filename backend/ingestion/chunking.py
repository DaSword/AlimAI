"""
Chunking Module - Semantic chunking for Islamic texts.

This module splits large text entries (Quran verses with Tafsir, Hadiths, etc.)
into smaller, meaningful chunks for efficient embedding generation and retrieval.

Migrated from quran_chunker.py with updated imports for backend structure.
Will be extended with LlamaIndex NodeParsers in Stage 4.
"""

import re
from typing import List, Dict, Optional
from html.parser import HTMLParser

# Updated imports for backend structure
from backend.core.config import Config
from backend.core.utils import setup_logging

logger = setup_logging("chunking")


class TafsirHTMLParser(HTMLParser):
    """Parse HTML tafsir content and extract text with semantic boundaries."""
    
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.current_text = []
        self.in_heading = False
        self.heading_level = 0
        
    def handle_starttag(self, tag, attrs):
        """Handle opening tags to identify semantic boundaries."""
        if tag in ['h1', 'h2', 'h3', 'h4']:
            # Flush current text before heading
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text:
                    self.chunks.append({
                        'text': text,
                        'is_heading': False
                    })
                self.current_text = []
            
            self.in_heading = True
            self.heading_level = int(tag[1])
        elif tag in ['p', 'div', 'br']:
            # Paragraph boundaries
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text:
                    self.chunks.append({
                        'text': text,
                        'is_heading': self.in_heading
                    })
                self.current_text = []
            self.in_heading = False
    
    def handle_endtag(self, tag):
        """Handle closing tags."""
        if tag in ['h1', 'h2', 'h3', 'h4']:
            if self.current_text:
                text = ' '.join(self.current_text).strip()
                if text:
                    self.chunks.append({
                        'text': text,
                        'is_heading': True
                    })
                self.current_text = []
            self.in_heading = False
    
    def handle_data(self, data):
        """Handle text data."""
        cleaned = data.strip()
        if cleaned:
            self.current_text.append(cleaned)
    
    def get_chunks(self):
        """Get all parsed chunks."""
        # Flush any remaining text
        if self.current_text:
            text = ' '.join(self.current_text).strip()
            if text:
                self.chunks.append({
                    'text': text,
                    'is_heading': self.in_heading
                })
        return self.chunks


def parse_tafsir_html(html: str, max_chunk_size: int = None, min_chunk_size: int = None) -> List[str]:
    """
    Parse HTML tafsir and split by semantic boundaries.
    
    Args:
        html: HTML content of the tafsir
        max_chunk_size: Maximum characters per chunk (defaults to config)
        min_chunk_size: Minimum characters per chunk (defaults to config)
        
    Returns:
        List of text chunks
    """
    if max_chunk_size is None:
        max_chunk_size = Config.CHUNK_SIZE_MAX
    if min_chunk_size is None:
        min_chunk_size = Config.CHUNK_SIZE_MIN
        
    if not html or not html.strip():
        return []
    
    # Parse HTML
    parser = TafsirHTMLParser()
    try:
        parser.feed(html)
    except Exception as e:
        # If HTML parsing fails, fall back to plain text
        logger.warning(f"HTML parsing failed: {e}, using plain text")
        return chunk_plain_text(html, max_chunk_size, min_chunk_size)
    
    semantic_chunks = parser.get_chunks()
    
    # Combine small chunks and split large ones
    final_chunks = []
    current_chunk = []
    current_size = 0
    
    for chunk in semantic_chunks:
        text = chunk['text']
        is_heading = chunk['is_heading']
        
        # If this is a heading and we have accumulated text, flush it
        if is_heading and current_chunk:
            combined = ' '.join(current_chunk).strip()
            if combined:
                final_chunks.append(combined)
            current_chunk = []
            current_size = 0
        
        # If adding this would exceed max size, flush current
        if current_size + len(text) > max_chunk_size and current_chunk:
            combined = ' '.join(current_chunk).strip()
            if combined:
                final_chunks.append(combined)
            current_chunk = []
            current_size = 0
        
        # If this single chunk is too large, split it
        if len(text) > max_chunk_size:
            # First, flush any current content
            if current_chunk:
                combined = ' '.join(current_chunk).strip()
                if combined:
                    final_chunks.append(combined)
                current_chunk = []
                current_size = 0
            
            # Split the large chunk
            split_chunks = split_large_text(text, max_chunk_size)
            final_chunks.extend(split_chunks)
        else:
            # Add to current chunk
            current_chunk.append(text)
            current_size += len(text) + 1  # +1 for space
    
    # Flush remaining
    if current_chunk:
        combined = ' '.join(current_chunk).strip()
        if combined:
            final_chunks.append(combined)
    
    # Filter out chunks that are too small (unless they're the only chunk)
    if len(final_chunks) > 1:
        final_chunks = [c for c in final_chunks if len(c) >= min_chunk_size]
    
    return final_chunks


def chunk_plain_text(text: str, max_chunk_size: int = None, min_chunk_size: int = None) -> List[str]:
    """
    Chunk plain text by sentences and paragraphs.
    
    Args:
        text: Plain text to chunk
        max_chunk_size: Maximum characters per chunk (defaults to config)
        min_chunk_size: Minimum characters per chunk (defaults to config)
        
    Returns:
        List of text chunks
    """
    if max_chunk_size is None:
        max_chunk_size = Config.CHUNK_SIZE_MAX
    if min_chunk_size is None:
        min_chunk_size = Config.CHUNK_SIZE_MIN
        
    if not text or not text.strip():
        return []
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split by paragraphs first (double newlines or similar patterns)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if current_size + len(para) > max_chunk_size and current_chunk:
            # Flush current chunk
            combined = ' '.join(current_chunk).strip()
            if combined:
                chunks.append(combined)
            current_chunk = []
            current_size = 0
        
        if len(para) > max_chunk_size:
            # Para is too large, split by sentences
            sentences = split_into_sentences(para)
            for sent in sentences:
                if current_size + len(sent) > max_chunk_size and current_chunk:
                    combined = ' '.join(current_chunk).strip()
                    if combined:
                        chunks.append(combined)
                    current_chunk = []
                    current_size = 0
                
                if len(sent) > max_chunk_size:
                    # Even sentence is too large, hard split
                    split_chunks = split_large_text(sent, max_chunk_size)
                    chunks.extend(split_chunks)
                else:
                    current_chunk.append(sent)
                    current_size += len(sent) + 1
        else:
            current_chunk.append(para)
            current_size += len(para) + 1
    
    # Flush remaining
    if current_chunk:
        combined = ' '.join(current_chunk).strip()
        if combined:
            chunks.append(combined)
    
    return chunks


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_large_text(text: str, max_size: int) -> List[str]:
    """Hard split large text into chunks."""
    chunks = []
    for i in range(0, len(text), max_size):
        chunk = text[i:i + max_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def create_chunk_metadata(
    verse_data: Dict,
    chunk_text: str,
    chunk_type: str,
    chunk_index: int,
    tafsir_source: Optional[str] = None
) -> Dict:
    """
    Create full metadata for a chunk.
    
    Args:
        verse_data: Original verse data
        chunk_text: Text content of the chunk
        chunk_type: "verse" or "tafsir"
        chunk_index: Index of chunk within verse
        tafsir_source: Source of tafsir (e.g., "ibn_kathir") if applicable
        
    Returns:
        Dictionary with chunk metadata
    """
    metadata = {
        'verse_key': verse_data.get('verse_key', ''),
        'chapter_number': verse_data.get('chapter_number'),
        'verse_number': verse_data.get('verse_number'),
        'chapter_name': verse_data.get('chapter_name'),
        'chapter_name_arabic': verse_data.get('chapter_name_arabic'),
        'arabic_text': verse_data.get('arabic_text'),
        'source': verse_data.get('source', 'Quran'),
        'source_detail': verse_data.get('source_detail', ''),
        'metadata': verse_data.get('metadata', {}),
        'chunk_type': chunk_type,
        'chunk_index': chunk_index,
        'text_content': chunk_text,
    }
    
    if tafsir_source:
        metadata['tafsir_source'] = tafsir_source
    
    # Include english_text for verse chunks
    if chunk_type == 'verse':
        metadata['english_text'] = verse_data.get('english_text', '')
    
    return metadata


def chunk_verse(
    verse_data: Dict,
    max_chunk_size: int = None,
    min_chunk_size: int = None,
    split_verses: bool = False,
    split_tafsirs: bool = True
) -> List[Dict]:
    """
    Split a verse entry into multiple semantic chunks.
    
    Args:
        verse_data: Dictionary containing verse data
        max_chunk_size: Maximum characters per chunk (defaults to config)
        min_chunk_size: Minimum characters per chunk (defaults to config)
        split_verses: Whether to split verse text itself
        split_tafsirs: Whether to split tafsir commentaries
        
    Returns:
        List of chunk dictionaries, each with metadata and text_content
    """
    if max_chunk_size is None:
        max_chunk_size = Config.CHUNK_SIZE_MAX
    if min_chunk_size is None:
        min_chunk_size = Config.CHUNK_SIZE_MIN
        
    chunks = []
    chunk_index = 0
    
    # 1. Create verse chunk
    verse_text = f"Surah {verse_data.get('chapter_name', '')} ({verse_data.get('chapter_name_arabic', '')}), Verse {verse_data.get('verse_number', '')}\n"
    verse_text += f"Arabic: {verse_data.get('arabic_text', '')}\n"
    verse_text += f"English: {verse_data.get('english_text', '')}"
    
    if split_verses and len(verse_text) > max_chunk_size:
        # Split long verses (rare case)
        verse_chunks = chunk_plain_text(verse_text, max_chunk_size, min_chunk_size)
        for v_chunk in verse_chunks:
            chunks.append(create_chunk_metadata(
                verse_data, v_chunk, 'verse', chunk_index
            ))
            chunk_index += 1
    else:
        # Single verse chunk
        chunks.append(create_chunk_metadata(
            verse_data, verse_text, 'verse', chunk_index
        ))
        chunk_index += 1
    
    # 2. Process tafsirs if present
    if split_tafsirs and 'tafsirs' in verse_data:
        tafsirs = verse_data['tafsirs']
        
        if isinstance(tafsirs, dict):
            # Multiple tafsir sources
            for source, tafsir_html in tafsirs.items():
                if not tafsir_html or not tafsir_html.strip():
                    continue
                
                # Parse and chunk this tafsir
                tafsir_chunks = parse_tafsir_html(tafsir_html, max_chunk_size, min_chunk_size)
                
                for t_chunk in tafsir_chunks:
                    # Add source label to beginning of first chunk
                    if t_chunk == tafsir_chunks[0]:
                        t_chunk = f"[Tafsir - {source}] {t_chunk}"
                    
                    chunks.append(create_chunk_metadata(
                        verse_data, t_chunk, 'tafsir', chunk_index, source
                    ))
                    chunk_index += 1
        
        elif isinstance(tafsirs, str):
            # Single tafsir string
            tafsir_chunks = parse_tafsir_html(tafsirs, max_chunk_size, min_chunk_size)
            for t_chunk in tafsir_chunks:
                chunks.append(create_chunk_metadata(
                    verse_data, t_chunk, 'tafsir', chunk_index, 'unknown'
                ))
                chunk_index += 1
    
    return chunks


def main():
    """Test the chunking functionality."""
    import json
    from backend.config import Config
    
    # Load a sample verse
    quran_file = Config.get_data_path('quran.json')
    
    if not quran_file.exists():
        print(f"Quran data file not found: {quran_file}")
        return
    
    with open(quran_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    verses = data['verses']
    
    # Test with first verse (which has a lot of tafsir)
    test_verse = verses[0]
    
    print(f"Testing chunking for verse: {test_verse['verse_key']}")
    print(f"Original text_for_embedding length: {len(test_verse.get('text_for_embedding', ''))}")
    print()
    
    # Generate chunks
    chunks = chunk_verse(test_verse)
    
    print(f"Generated {len(chunks)} chunks:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        print(f"  Type: {chunk['chunk_type']}")
        print(f"  Index: {chunk['chunk_index']}")
        if chunk.get('tafsir_source'):
            print(f"  Tafsir Source: {chunk['tafsir_source']}")
        print(f"  Length: {len(chunk['text_content'])} chars")
        print(f"  Preview: {chunk['text_content'][:200]}...")
        print("-" * 60)
    
    # Stats
    verse_chunks = [c for c in chunks if c['chunk_type'] == 'verse']
    tafsir_chunks = [c for c in chunks if c['chunk_type'] == 'tafsir']
    
    print("\nSummary:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Verse chunks: {len(verse_chunks)}")
    print(f"  Tafsir chunks: {len(tafsir_chunks)}")
    print(f"  Avg chunk size: {sum(len(c['text_content']) for c in chunks) / len(chunks):.0f} chars")


if __name__ == "__main__":
    main()

