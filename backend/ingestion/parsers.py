"""
LlamaIndex NodeParsers for Islamic Texts - Stage 4 Implementation

This module provides custom NodeParsers for different types of Islamic texts:
- QuranNodeParser: Quran verses with tafsir
- HadithNodeParser: Individual hadiths with chain of narration
- TafsirNodeParser: Verse-by-verse commentary
- FiqhNodeParser: Fiqh rulings by topic and madhab
- SeerahNodeParser: Chronological events from the Prophet's biography

Each parser extends LlamaIndex's NodeParser base class and creates properly
structured nodes with metadata following our universal schema.
"""

from typing import List, Dict, Any, Optional, Sequence
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import Document, TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.utils import get_tqdm_iterable
import uuid

from backend.core.config import Config
from backend.core.models import SourceType, ChunkType, create_qdrant_payload
from backend.ingestion.chunking import parse_tafsir_html, chunk_plain_text
from backend.core.utils import setup_logging

logger = setup_logging("parsers")


class QuranNodeParser(NodeParser):
    """
    NodeParser for Quran verses with tafsir commentary.
    
    Creates separate nodes for:
    - Verse text (Arabic + English)
    - Tafsir commentary chunks (if present)
    
    Each node includes proper metadata for source tracking and cross-referencing.
    """
    
    # Store config as class attributes to avoid Pydantic field issues
    _max_chunk_size: int
    _min_chunk_size: int
    _split_verses: bool
    _split_tafsirs: bool
    
    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        split_verses: bool = False,
        split_tafsirs: bool = True,
        include_metadata: bool = True,
    ):
        """
        Initialize QuranNodeParser.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            split_verses: Whether to split verse text itself (usually False)
            split_tafsirs: Whether to split tafsir commentaries (usually True)
            include_metadata: Include detailed metadata in nodes
        """
        super().__init__(
            include_metadata=include_metadata,
        )
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_max_chunk_size', max_chunk_size or Config.CHUNK_SIZE_MAX)
        object.__setattr__(self, '_min_chunk_size', min_chunk_size or Config.CHUNK_SIZE_MIN)
        object.__setattr__(self, '_split_verses', split_verses)
        object.__setattr__(self, '_split_tafsirs', split_tafsirs)
    
    def _parse_nodes(
        self,
        nodes: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[TextNode]:
        """
        Parse Quran documents into TextNodes.
        
        Args:
            nodes: List of Document objects containing Quran verse data
            show_progress: Show progress bar
            
        Returns:
            List of TextNode objects
        """
        all_nodes: List[TextNode] = []
        
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing Quran verses"
        )
        
        for doc in nodes_with_progress:
            # Extract verse data from document
            verse_data = doc.metadata
            
            # Generate nodes for this verse
            verse_nodes = self._chunk_verse(doc, verse_data)
            all_nodes.extend(verse_nodes)
        
        return all_nodes
    
    def _chunk_verse(self, doc: Document, verse_data: Dict[str, Any]) -> List[TextNode]:
        """
        Split a single verse entry into multiple semantic chunks.
        
        Args:
            doc: Document object containing verse data
            verse_data: Dictionary with verse metadata
            
        Returns:
            List of TextNode objects
        """
        chunks: List[TextNode] = []
        chunk_index = 0
        
        # Extract verse information
        verse_key = verse_data.get('verse_key', '')
        surah_number = verse_data.get('chapter_number', 0)
        verse_number = verse_data.get('verse_number', 0)
        surah_name = verse_data.get('chapter_name', '')
        surah_name_arabic = verse_data.get('chapter_name_arabic', '')
        arabic_text = verse_data.get('arabic_text', '')
        english_text = verse_data.get('english_text', '')
        
        # 1. Create verse node
        verse_text = f"Surah {surah_name} ({surah_name_arabic}), Verse {verse_number}\n"
        verse_text += f"Arabic: {arabic_text}\n"
        verse_text += f"English: {english_text}"
        
        # Create verse node metadata
        verse_payload = create_qdrant_payload(
            source_type=SourceType.QURAN,
            book_title="The Noble Quran",
            book_title_arabic="القرآن الكريم",
            author="Allah (Revealed)",
            text_content=verse_text,
            arabic_text=arabic_text,
            english_text=english_text,
            topic_tags=[],  # Can be populated later
            # Source metadata
            surah_number=surah_number,
            verse_number=verse_number,
            verse_key=verse_key,
            surah_name=surah_name,
            surah_name_arabic=surah_name_arabic,
            chunk_type=ChunkType.VERSE,
            chunk_index=chunk_index,
        )
        
        # Create verse TextNode
        verse_node = TextNode(
            id_=str(uuid.uuid4()),
            text=verse_text,
            metadata=verse_payload.dict(exclude_none=True),
        )
        
        # Link to source document
        verse_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc.id_ or doc.doc_id,
        )
        
        chunks.append(verse_node)
        chunk_index += 1
        
        # 2. Process tafsirs if present and splitting is enabled
        if self._split_tafsirs and 'tafsirs' in verse_data:
            tafsirs = verse_data['tafsirs']
            
            if isinstance(tafsirs, dict):
                # Multiple tafsir sources
                for source, tafsir_html in tafsirs.items():
                    if not tafsir_html or not tafsir_html.strip():
                        continue
                    
                    # Parse and chunk this tafsir
                    tafsir_chunks = parse_tafsir_html(
                        tafsir_html, 
                        self._max_chunk_size, 
                        self._min_chunk_size
                    )
                    
                    for t_chunk in tafsir_chunks:
                        # Add source label to beginning of first chunk
                        if t_chunk == tafsir_chunks[0]:
                            t_chunk = f"[Tafsir - {source}] {t_chunk}"
                        
                        # Create tafsir node metadata
                        tafsir_payload = create_qdrant_payload(
                            source_type=SourceType.TAFSIR,
                            book_title=f"Tafsir {source}",
                            author=source,
                            text_content=t_chunk,
                            topic_tags=[],
                            # Source metadata
                            surah_number=surah_number,
                            verse_number=verse_number,
                            verse_key=verse_key,
                            surah_name=surah_name,
                            chunk_type=ChunkType.TAFSIR,
                            chunk_index=chunk_index,
                            tafsir_source=source,
                            # References
                            related_verses=[verse_key],
                        )
                        
                        # Create tafsir TextNode
                        tafsir_node = TextNode(
                            id_=str(uuid.uuid4()),
                            text=t_chunk,
                            metadata=tafsir_payload.dict(exclude_none=True),
                        )
                        
                        # Link to verse node
                        tafsir_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                            node_id=verse_node.id_ or verse_node.node_id,
                        )
                        
                        chunks.append(tafsir_node)
                        chunk_index += 1
            
            elif isinstance(tafsirs, str):
                # Single tafsir string
                tafsir_chunks = parse_tafsir_html(
                    tafsirs, 
                    self._max_chunk_size, 
                    self._min_chunk_size
                )
                
                for t_chunk in tafsir_chunks:
                    tafsir_payload = create_qdrant_payload(
                        source_type=SourceType.TAFSIR,
                        book_title="Tafsir (unknown source)",
                        author="Unknown",
                        text_content=t_chunk,
                        topic_tags=[],
                        surah_number=surah_number,
                        verse_number=verse_number,
                        verse_key=verse_key,
                        surah_name=surah_name,
                        chunk_type=ChunkType.TAFSIR,
                        chunk_index=chunk_index,
                        tafsir_source='unknown',
                        related_verses=[verse_key],
                    )
                    
                    tafsir_node = TextNode(
                        id_=str(uuid.uuid4()),
                        text=t_chunk,
                        metadata=tafsir_payload.dict(exclude_none=True),
                    )
                    
                    tafsir_node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                        node_id=verse_node.id_ or verse_node.node_id,
                    )
                    
                    chunks.append(tafsir_node)
                    chunk_index += 1
        
        return chunks


class HadithNodeParser(NodeParser):
    """
    NodeParser for Hadith collections.
    
    Creates nodes for individual hadiths with:
    - Hadith number and book/chapter metadata
    - Isnad (chain of narration)
    - Matn (hadith text)
    - Authenticity grade
    """
    
    _max_chunk_size: int
    
    def __init__(
        self,
        max_chunk_size: int = None,
        include_metadata: bool = True,
    ):
        super().__init__(include_metadata=include_metadata)
        object.__setattr__(self, '_max_chunk_size', max_chunk_size or Config.CHUNK_SIZE_MAX)
    
    def _parse_nodes(
        self,
        nodes: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[TextNode]:
        """Parse Hadith documents into TextNodes."""
        all_nodes: List[TextNode] = []
        
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing Hadiths"
        )
        
        for doc in nodes_with_progress:
            hadith_data = doc.metadata
            hadith_node = self._create_hadith_node(doc, hadith_data)
            all_nodes.append(hadith_node)
        
        return all_nodes
    
    def _create_hadith_node(self, doc: Document, hadith_data: Dict[str, Any]) -> TextNode:
        """Create a TextNode for a single hadith."""
        
        # Extract hadith information
        book_title = hadith_data.get('book_title', 'Unknown Hadith Collection')
        hadith_number = hadith_data.get('hadith_number', 0)
        book_name = hadith_data.get('book_name', '')
        chapter_name = hadith_data.get('chapter_name', '')
        narrator_chain = hadith_data.get('narrator_chain', '')
        hadith_text_arabic = hadith_data.get('arabic_text', '')
        hadith_text_english = hadith_data.get('english_text', '')
        authenticity = hadith_data.get('authenticity_grade', 'unknown')
        
        # Build hadith text content
        text_content = f"{book_title} - Hadith #{hadith_number}\n"
        if book_name:
            text_content += f"Book: {book_name}\n"
        if chapter_name:
            text_content += f"Chapter: {chapter_name}\n"
        if narrator_chain:
            text_content += f"Chain: {narrator_chain}\n"
        if hadith_text_arabic:
            text_content += f"Arabic: {hadith_text_arabic}\n"
        if hadith_text_english:
            text_content += f"English: {hadith_text_english}"
        
        # Create payload with universal schema
        payload = create_qdrant_payload(
            source_type=SourceType.HADITH,
            book_title=book_title,
            book_title_arabic=hadith_data.get('book_title_arabic', ''),
            author=hadith_data.get('author', 'Unknown'),
            text_content=text_content,
            arabic_text=hadith_text_arabic,
            english_text=hadith_text_english,
            topic_tags=hadith_data.get('topic_tags', []),
            # Hadith-specific metadata
            hadith_number=hadith_number,
            book_name=book_name,
            chapter_name=chapter_name,
            narrator_chain=narrator_chain,
            authenticity_grade=authenticity,
            # References
            related_verses=hadith_data.get('related_verses', []),
            related_topics=hadith_data.get('related_topics', []),
        )
        
        # Create TextNode
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=text_content,
            metadata=payload.dict(exclude_none=True),
        )
        
        # Link to source document
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc.id_ or doc.doc_id,
        )
        
        return node


class TafsirNodeParser(NodeParser):
    """
    NodeParser for standalone Tafsir texts (not embedded in Quran data).
    
    Creates nodes for verse-by-verse commentary with proper cross-references.
    """
    
    _max_chunk_size: int
    _min_chunk_size: int
    
    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        include_metadata: bool = True,
    ):
        super().__init__(include_metadata=include_metadata)
        object.__setattr__(self, '_max_chunk_size', max_chunk_size or Config.CHUNK_SIZE_MAX)
        object.__setattr__(self, '_min_chunk_size', min_chunk_size or Config.CHUNK_SIZE_MIN)
    
    def _parse_nodes(
        self,
        nodes: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[TextNode]:
        """Parse Tafsir documents into TextNodes."""
        all_nodes: List[TextNode] = []
        
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing Tafsir"
        )
        
        for doc in nodes_with_progress:
            tafsir_data = doc.metadata
            
            # Parse tafsir text (might be HTML)
            tafsir_text = doc.text or tafsir_data.get('tafsir_text', '')
            
            if tafsir_data.get('is_html', False):
                chunks = parse_tafsir_html(tafsir_text, self._max_chunk_size, self._min_chunk_size)
            else:
                chunks = chunk_plain_text(tafsir_text, self._max_chunk_size, self._min_chunk_size)
            
            # Create nodes for each chunk
            for idx, chunk in enumerate(chunks):
                tafsir_node = self._create_tafsir_node(doc, tafsir_data, chunk, idx)
                all_nodes.append(tafsir_node)
        
        return all_nodes
    
    def _create_tafsir_node(
        self, 
        doc: Document, 
        tafsir_data: Dict[str, Any],
        chunk_text: str,
        chunk_index: int
    ) -> TextNode:
        """Create a TextNode for a tafsir chunk."""
        
        verse_key = tafsir_data.get('verse_key', '')
        tafsir_source = tafsir_data.get('tafsir_source', 'Unknown')
        book_title = tafsir_data.get('book_title', f'Tafsir {tafsir_source}')
        
        payload = create_qdrant_payload(
            source_type=SourceType.TAFSIR,
            book_title=book_title,
            author=tafsir_data.get('author', tafsir_source),
            text_content=chunk_text,
            topic_tags=tafsir_data.get('topic_tags', []),
            # Tafsir-specific metadata
            verse_key=verse_key,
            surah_number=tafsir_data.get('surah_number'),
            verse_number=tafsir_data.get('verse_number'),
            surah_name=tafsir_data.get('surah_name'),
            tafsir_source=tafsir_source,
            chunk_type=ChunkType.TAFSIR,
            chunk_index=chunk_index,
            # References
            related_verses=[verse_key] if verse_key else [],
        )
        
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=chunk_text,
            metadata=payload.dict(exclude_none=True),
        )
        
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc.id_ or doc.doc_id,
        )
        
        return node


class FiqhNodeParser(NodeParser):
    """
    NodeParser for Fiqh (Islamic jurisprudence) texts.
    
    Creates nodes for rulings with:
    - Madhab (school of thought)
    - Ruling category (prayer, fasting, etc.)
    - Hierarchical structure (Book → Chapter → Ruling)
    """
    
    _max_chunk_size: int
    _min_chunk_size: int
    
    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        include_metadata: bool = True,
    ):
        super().__init__(include_metadata=include_metadata)
        object.__setattr__(self, '_max_chunk_size', max_chunk_size or Config.CHUNK_SIZE_MAX)
        object.__setattr__(self, '_min_chunk_size', min_chunk_size or Config.CHUNK_SIZE_MIN)
    
    def _parse_nodes(
        self,
        nodes: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[TextNode]:
        """Parse Fiqh documents into TextNodes."""
        all_nodes: List[TextNode] = []
        
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing Fiqh rulings"
        )
        
        for doc in nodes_with_progress:
            fiqh_data = doc.metadata
            
            # Chunk ruling text if needed
            ruling_text = doc.text or fiqh_data.get('ruling_text', '')
            chunks = chunk_plain_text(ruling_text, self._max_chunk_size, self._min_chunk_size)
            
            # Create nodes for each chunk
            for idx, chunk in enumerate(chunks):
                fiqh_node = self._create_fiqh_node(doc, fiqh_data, chunk, idx)
                all_nodes.append(fiqh_node)
        
        return all_nodes
    
    def _create_fiqh_node(
        self,
        doc: Document,
        fiqh_data: Dict[str, Any],
        chunk_text: str,
        chunk_index: int
    ) -> TextNode:
        """Create a TextNode for a fiqh ruling chunk."""
        
        book_title = fiqh_data.get('book_title', 'Unknown Fiqh Text')
        madhab = fiqh_data.get('madhab', 'unknown')
        ruling_category = fiqh_data.get('ruling_category', 'general')
        
        payload = create_qdrant_payload(
            source_type=SourceType.FIQH,
            book_title=book_title,
            book_title_arabic=fiqh_data.get('book_title_arabic', ''),
            author=fiqh_data.get('author', 'Unknown'),
            text_content=chunk_text,
            topic_tags=fiqh_data.get('topic_tags', []),
            # Fiqh-specific metadata
            madhab=madhab,
            ruling_category=ruling_category,
            # References
            related_verses=fiqh_data.get('related_verses', []),
            related_hadiths=fiqh_data.get('related_hadiths', []),
            related_topics=fiqh_data.get('related_topics', []),
        )
        
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=chunk_text,
            metadata=payload.dict(exclude_none=True),
        )
        
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc.id_ or doc.doc_id,
        )
        
        return node


class SeerahNodeParser(NodeParser):
    """
    NodeParser for Seerah (Prophet's biography) texts.
    
    Creates nodes for chronological events with:
    - Event name
    - Hijri year
    - Location
    - Participants
    """
    
    _max_chunk_size: int
    _min_chunk_size: int
    
    def __init__(
        self,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        include_metadata: bool = True,
    ):
        super().__init__(include_metadata=include_metadata)
        object.__setattr__(self, '_max_chunk_size', max_chunk_size or Config.CHUNK_SIZE_MAX)
        object.__setattr__(self, '_min_chunk_size', min_chunk_size or Config.CHUNK_SIZE_MIN)
    
    def _parse_nodes(
        self,
        nodes: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[TextNode]:
        """Parse Seerah documents into TextNodes."""
        all_nodes: List[TextNode] = []
        
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Parsing Seerah events"
        )
        
        for doc in nodes_with_progress:
            seerah_data = doc.metadata
            
            # Chunk event text if needed
            event_text = doc.text or seerah_data.get('event_text', '')
            chunks = chunk_plain_text(event_text, self._max_chunk_size, self._min_chunk_size)
            
            # Create nodes for each chunk
            for idx, chunk in enumerate(chunks):
                seerah_node = self._create_seerah_node(doc, seerah_data, chunk, idx)
                all_nodes.append(seerah_node)
        
        return all_nodes
    
    def _create_seerah_node(
        self,
        doc: Document,
        seerah_data: Dict[str, Any],
        chunk_text: str,
        chunk_index: int
    ) -> TextNode:
        """Create a TextNode for a seerah event chunk."""
        
        book_title = seerah_data.get('book_title', 'Al-Sirah al-Nabawiyyah')
        event_name = seerah_data.get('event_name', 'Unknown Event')
        
        payload = create_qdrant_payload(
            source_type=SourceType.SEERAH,
            book_title=book_title,
            book_title_arabic=seerah_data.get('book_title_arabic', ''),
            author=seerah_data.get('author', 'Ibn Hisham'),
            text_content=chunk_text,
            topic_tags=seerah_data.get('topic_tags', []),
            # Seerah-specific metadata
            event_name=event_name,
            chronological_order=seerah_data.get('chronological_order'),
            year_hijri=seerah_data.get('year_hijri'),
            # References
            related_verses=seerah_data.get('related_verses', []),
            related_hadiths=seerah_data.get('related_hadiths', []),
            related_topics=seerah_data.get('related_topics', []),
        )
        
        node = TextNode(
            id_=str(uuid.uuid4()),
            text=chunk_text,
            metadata=payload.dict(exclude_none=True),
        )
        
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=doc.id_ or doc.doc_id,
        )
        
        return node


# Export all parsers
__all__ = [
    'QuranNodeParser',
    'HadithNodeParser',
    'TafsirNodeParser',
    'FiqhNodeParser',
    'SeerahNodeParser',
]

