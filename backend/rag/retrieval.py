"""
LlamaIndex query engine wrapper for multi-source retrieval.

This module provides:
- Query engine wrapper with metadata filtering
- Multi-query expansion
- Source-specific retrieval strategies
- Madhab-aware fiqh retrieval
- Cross-reference resolution
"""

from typing import List, Optional, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.core.schema import NodeWithScore, TextNode

from backend.core.models import (
    DocumentChunk,
    SourceType,
    QuestionType,
    Madhab,
    AuthenticityGrade,
    QdrantPayload,
)
from backend.vectordb.qdrant_manager import QdrantManager
from backend.llama.llama_config import get_embed_model, get_llm
from backend.core.config import Config


class IslamicRetriever:
    """
    Intelligent retriever for Islamic knowledge base.
    
    Provides source-aware retrieval with metadata filtering,
    multi-query expansion, and authenticity-based ranking.
    """
    
    def __init__(
        self,
        collection_name: str = "islamic_knowledge",
        embedding_backend: Optional[str] = None,
        llm_backend: Optional[str] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Qdrant collection name
            embedding_backend: Embedding backend (huggingface/ollama/lmstudio)
            llm_backend: LLM backend (ollama/lmstudio)
        """
        self.config = Config()
        self.collection_name = collection_name
        
        # Get backends from config if not specified
        self.embedding_backend = embedding_backend or self.config.EMBEDDING_BACKEND
        self.llm_backend = llm_backend or self.config.LLM_BACKEND
        
        # Initialize Qdrant manager
        self.qdrant_manager = QdrantManager(
            collection_name=collection_name,
        )
        
        # Get LlamaIndex vector store
        self.vector_store = self.qdrant_manager.get_vector_store()
        print(f"Embedding backend: {self.embedding_backend}")
        
        # Create vector store index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=get_embed_model(embedding_backend=self.embedding_backend),
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[MetadataFilters] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve documents for a query with optional filters.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filters: Metadata filters
            
        Returns:
            List of retrieved document chunks
        """
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
            filters=filters,
        )
        
        # Retrieve nodes
        nodes = retriever.retrieve(query)
        
        # Convert to DocumentChunk
        return self._nodes_to_chunks(nodes)
    
    def retrieve_by_source_type(
        self,
        query: str,
        source_types: List[SourceType],
        top_k: int = 10,
    ) -> List[DocumentChunk]:
        """
        Retrieve documents filtered by source type.
        
        Args:
            query: User query
            source_types: List of source types to retrieve from
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        # Build filters
        if len(source_types) == 1:
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="source_type",
                        value=source_types[0].value,
                        operator=FilterOperator.EQ,
                    )
                ]
            )
        else:
            # Multiple source types - use IN operator
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="source_type",
                        value=[st.value for st in source_types],
                        operator=FilterOperator.IN,
                    )
                ]
            )
        
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def retrieve_by_madhab(
        self,
        query: str,
        madhab: Madhab,
        top_k: int = 5,
    ) -> List[DocumentChunk]:
        """
        Retrieve fiqh documents from a specific madhab.
        
        Args:
            query: User query
            madhab: School of jurisprudence
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source_type",
                    value="fiqh",
                    operator=FilterOperator.EQ,
                ),
                MetadataFilter(
                    key="source_metadata.madhab",
                    value=madhab.value,
                    operator=FilterOperator.EQ,
                ),
            ]
        )
        
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def retrieve_by_authenticity(
        self,
        query: str,
        min_grade: AuthenticityGrade = AuthenticityGrade.HASAN,
        top_k: int = 10,
    ) -> List[DocumentChunk]:
        """
        Retrieve hadith with minimum authenticity grade.
        
        Args:
            query: User query
            min_grade: Minimum authenticity grade
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        # Define grade hierarchy
        grade_hierarchy = {
            AuthenticityGrade.SAHIH: ["sahih"],
            AuthenticityGrade.HASAN: ["sahih", "hasan"],
            AuthenticityGrade.DAIF: ["sahih", "hasan", "daif"],
        }
        
        allowed_grades = grade_hierarchy.get(min_grade, ["sahih", "hasan"])
        
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source_type",
                    value="hadith",
                    operator=FilterOperator.EQ,
                ),
                MetadataFilter(
                    key="source_metadata.authenticity_grade",
                    value=allowed_grades,
                    operator=FilterOperator.IN,
                ),
            ]
        )
        
        return self.retrieve(query, top_k=top_k, filters=filters)
    
    def retrieve_quran_with_tafsir(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[DocumentChunk]:
        """
        Retrieve Quranic verses along with their tafsir.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks (verses + tafsir)
        """
        # Retrieve from both Quran and Tafsir sources
        results = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.QURAN, SourceType.TAFSIR],
            top_k=top_k,
        )
        
        return results
    
    def retrieve_for_question_type(
        self,
        query: str,
        question_type: QuestionType,
        top_k: int = 10,
        madhab_preference: Optional[Madhab] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve documents based on question type with intelligent filtering.
        
        Args:
            query: User query
            question_type: Type of question
            top_k: Number of documents to retrieve
            madhab_preference: Preferred madhab for fiqh questions
            
        Returns:
            List of retrieved document chunks
        """
        if question_type == QuestionType.FIQH:
            return self._retrieve_for_fiqh(query, top_k, madhab_preference)
        elif question_type == QuestionType.AQIDAH:
            return self._retrieve_for_aqidah(query, top_k)
        elif question_type == QuestionType.TAFSIR:
            return self.retrieve_quran_with_tafsir(query, top_k)
        elif question_type == QuestionType.HADITH:
            return self._retrieve_for_hadith(query, top_k)
        else:
            # General - retrieve from all sources
            return self.retrieve(query, top_k=top_k)
    
    def _retrieve_for_fiqh(
        self,
        query: str,
        top_k: int,
        madhab_preference: Optional[Madhab] = None,
    ) -> List[DocumentChunk]:
        """
        Retrieve for fiqh questions - get perspectives from multiple madhahib.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            madhab_preference: Preferred madhab
            
        Returns:
            List of retrieved document chunks
        """
        results = []
        
        # Get primary evidence from Quran and Hadith
        primary_sources = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.QURAN, SourceType.HADITH],
            top_k=top_k // 2,
        )
        results.extend(primary_sources)
        
        # Get fiqh rulings from multiple madhahib
        per_madhab = max(1, (top_k - len(primary_sources)) // 4)
        
        for madhab in [Madhab.HANAFI, Madhab.MALIKI, Madhab.SHAFI, Madhab.HANBALI]:
            madhab_results = self.retrieve_by_madhab(query, madhab, top_k=per_madhab)
            results.extend(madhab_results)
        
        # If madhab preference specified, boost those results
        if madhab_preference:
            preferred_results = self.retrieve_by_madhab(
                query, madhab_preference, top_k=per_madhab * 2
            )
            results = preferred_results + results
        
        return results[:top_k]
    
    def _retrieve_for_aqidah(
        self,
        query: str,
        top_k: int,
    ) -> List[DocumentChunk]:
        """
        Retrieve for aqidah questions - prioritize Quran and Sahih Hadith.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        results = []
        
        # Prioritize Quran
        quran_results = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.QURAN],
            top_k=top_k // 3,
        )
        results.extend(quran_results)
        
        # Get Sahih Hadith
        hadith_results = self.retrieve_by_authenticity(
            query=query,
            min_grade=AuthenticityGrade.SAHIH,
            top_k=top_k // 3,
        )
        results.extend(hadith_results)
        
        # Get aqidah texts
        aqidah_results = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.AQIDAH],
            top_k=top_k // 3,
        )
        results.extend(aqidah_results)

        print(f"Aqidah results: {aqidah_results}")
        
        return results[:top_k]
    
    def _retrieve_for_hadith(
        self,
        query: str,
        top_k: int,
    ) -> List[DocumentChunk]:
        """
        Retrieve for hadith questions - get hadith with related verses.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved document chunks
        """
        results = []
        
        # Get hadith (prioritize authentic)
        hadith_results = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.HADITH],
            top_k=int(top_k * 0.7),
        )
        results.extend(hadith_results)
        
        # Get related Quranic verses
        quran_results = self.retrieve_by_source_type(
            query=query,
            source_types=[SourceType.QURAN],
            top_k=int(top_k * 0.3),
        )
        results.extend(quran_results)
        
        return results[:top_k]
    
    def multi_query_retrieve(
        self,
        queries: List[str],
        top_k: int = 10,
        deduplicate: bool = True,
    ) -> List[DocumentChunk]:
        """
        Retrieve using multiple query variations.
        
        Args:
            queries: List of query variations
            top_k: Number of documents to retrieve per query
            deduplicate: Whether to remove duplicates
            
        Returns:
            Combined list of retrieved document chunks
        """
        all_results = []
        seen_ids = set()
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            
            if deduplicate:
                for doc in results:
                    if doc.id not in seen_ids:
                        all_results.append(doc)
                        seen_ids.add(doc.id)
            else:
                all_results.extend(results)
        
        return all_results
    
    def _nodes_to_chunks(self, nodes: List[NodeWithScore]) -> List[DocumentChunk]:
        """
        Convert LlamaIndex nodes to DocumentChunk objects.
        
        Args:
            nodes: List of NodeWithScore from LlamaIndex
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        for node_with_score in nodes:
            node = node_with_score.node
            score = node_with_score.score
            
            # Extract metadata
            metadata_dict = node.metadata if hasattr(node, 'metadata') else {}
            
            # Create QdrantPayload from metadata
            try:
                payload = QdrantPayload(**metadata_dict)
            except Exception as e:
                # Fallback to basic payload if validation fails
                payload = QdrantPayload(
                    source_type=metadata_dict.get("source_type", "unknown"),
                    book_title=metadata_dict.get("book_title", "Unknown"),
                    author=metadata_dict.get("author", "Unknown"),
                    text_content=node.get_content() if hasattr(node, 'get_content') else str(node),
                )
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                id=node.node_id if hasattr(node, 'node_id') else str(hash(node)),
                text_content=node.get_content() if hasattr(node, 'get_content') else str(node),
                metadata=payload,
                score=score,
            )
            
            chunks.append(chunk)
        
        return chunks


# ============================================================================
# Utility Functions
# ============================================================================

def create_retriever(
    collection_name: str = "islamic_knowledge",
    embedding_backend: Optional[str] = None,
    llm_backend: Optional[str] = None,
) -> IslamicRetriever:
    """
    Factory function to create an IslamicRetriever instance.
    
    Args:
        collection_name: Qdrant collection name
        embedding_backend: Embedding backend
        llm_backend: LLM backend
        
    Returns:
        IslamicRetriever instance
    """
    return IslamicRetriever(
        collection_name=collection_name,
        embedding_backend=embedding_backend,
        llm_backend=llm_backend,
    )


def expand_query_with_llm(
    query: str,
    question_type: QuestionType,
    llm_backend: Optional[str] = None,
    num_expansions: int = 2,
) -> List[str]:
    """
    Expand query using LLM for better retrieval coverage.
    
    Args:
        query: Original user query
        question_type: Type of question
        llm_backend: LLM backend to use
        num_expansions: Number of expanded queries to generate
        
    Returns:
        List of expanded query strings
    """
    from backend.rag.prompts import get_expansion_prompt, format_prompt
    
    # Get appropriate expansion prompt
    expansion_prompt = get_expansion_prompt(question_type)
    
    # Format prompt
    formatted_prompt = format_prompt(
        expansion_prompt,
        user_query=query,
        question_type=str(question_type),
    )
    
    # Get LLM
    llm = get_llm()
    
    # Generate expansions
    try:
        response = llm.complete(formatted_prompt)
        
        # Parse response (expecting Python list format)
        response_text = str(response)
        
        # Simple parsing: extract lines that look like questions
        expanded_queries = []
        for line in response_text.split('\n'):
            line = line.strip()
            # Remove numbering and quotes
            line = line.lstrip('0123456789.- ').strip('"\'')
            if line and len(line) > 10 and '?' in line:
                expanded_queries.append(line)
        
        # Return original query + expansions
        return [query] + expanded_queries[:num_expansions]
    
    except Exception as e:
        # Fallback: return original query
        print(f"Query expansion failed: {e}")
        return [query]

