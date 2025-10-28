"""
Reranker Service - Rerank retrieved documents for improved relevance.

This module provides reranking functionality using LlamaIndex's LLMRerank
postprocessor with LLM backends (Ollama or LM Studio). Follows LlamaIndex best practices.

Based on: https://developers.llamaindex.ai/python/examples/workflow/rag/

Created in Stage 3 for enhanced retrieval (optional).
"""

import time
from typing import List, Dict, Any, Optional

from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.schema import NodeWithScore

from backend.core.config import config
from backend.core.utils import setup_logging

logger = setup_logging("reranker_service")


class RerankerService:
    """
    Service class for reranking retrieved documents using LlamaIndex's LLMRerank.
    
    Supports multiple backends:
    - Ollama (default)
    - LM Studio (local server)
    
    The LLM evaluates query-document relevance to reorder results.
    """
    
    def __init__(
        self,
        llm_backend: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_n: int = 5,
        choice_batch_size: int = 10
    ):
        """
        Initialize RerankerService.
        
        Args:
            llm_backend: LLM backend ('ollama' or 'lmstudio', defaults to config)
            llm_model: LLM model name for reranking (defaults to config based on backend)
            top_n: Number of top results to return after reranking
            choice_batch_size: Batch size for reranking choices
        """
        self.llm_backend = llm_backend or config.LLM_BACKEND
        
        # Set model name based on backend
        if self.llm_backend == "lmstudio":
            self.llm_model = llm_model or config.LMSTUDIO_RERANKER_MODEL
        else:  # ollama
            self.llm_model = llm_model or config.OLLAMA_RERANKER_MODEL
        
        self.top_n = top_n
        self.choice_batch_size = choice_batch_size
        self.reranker = None
        
        logger.info("Initialized RerankerService")
        logger.info(f"  Backend: {self.llm_backend}")
        logger.info(f"  LLM Model: {self.llm_model}")
        logger.info(f"  Top N: {self.top_n}")
        logger.info(f"  Batch Size: {self.choice_batch_size}")
        
    def _get_reranker(self) -> LLMRerank:
        """
        Get or create the LLMRerank instance.
        
        Returns:
            Configured LLMRerank postprocessor
        """
        if self.reranker is None:
            logger.info(f"Creating LLMRerank with {self.llm_model} ({self.llm_backend})")
            
            # Create LLM instance based on backend
            if self.llm_backend == "ollama":
                from llama_index.llms.ollama import Ollama
                llm = Ollama(
                    model=self.llm_model,
                    base_url=config.OLLAMA_URL,
                    request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
                    context_window=config.OLLAMA_MAX_TOKENS,
                )
            elif self.llm_backend == "lmstudio":
                from llama_index.llms.lmstudio import LMStudio
                llm = LMStudio(
                    model_name=self.llm_model,
                    base_url=config.LMSTUDIO_URL,
                )
            else:
                raise ValueError(f"Unknown LLM backend: {self.llm_backend}")
            
            # Create LLMRerank postprocessor
            self.reranker = LLMRerank(
                llm=llm,
                top_n=self.top_n,
                choice_batch_size=self.choice_batch_size,
            )
            
            logger.info("LLMRerank ready")
        
        return self.reranker
    
    def check_availability(self) -> bool:
        """
        Check if the reranker can be initialized.
        
        Returns:
            True if reranker is available
        """
        try:
            self._get_reranker()
            return True
        except Exception as e:
            logger.error(f"Reranker not available: {str(e)}")
            return False
    
    def rerank(
        self,
        nodes: List[NodeWithScore],
        query_str: str
    ) -> List[NodeWithScore]:
        """
        Rerank nodes using LLM-based reranking.
        
        This uses LlamaIndex's LLMRerank postprocessor which uses an LLM
        to evaluate the relevance of each node to the query.
        
        Args:
            nodes: List of NodeWithScore objects from retrieval
            query_str: The original query string
            
        Returns:
            Reranked list of NodeWithScore objects (top_n results)
        """
        if not nodes:
            logger.warning("No nodes to rerank")
            return []
        
        try:
            reranker = self._get_reranker()
            
            logger.info(f"Reranking {len(nodes)} nodes with LLM...")
            start_time = time.time()
            
            # Use LlamaIndex's postprocess_nodes method
            reranked_nodes = reranker.postprocess_nodes(
                nodes,
                query_str=query_str
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Reranked to top {len(reranked_nodes)} nodes in {elapsed:.2f}s")
            
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            logger.warning("Returning original nodes without reranking")
            return nodes[:self.top_n]  # Just return top N original nodes
    
    def test_reranking(self) -> Dict[str, Any]:
        """
        Test the reranking with sample data.
        
        Returns:
            Dictionary with test results
        """
        from llama_index.core.schema import TextNode
        
        query = "What is Ramadan fasting?"
        
        # Create sample nodes (LlamaIndex format)
        nodes = [
            NodeWithScore(
                node=TextNode(text="Muslims fast during Ramadan from dawn to sunset."),
                score=0.8
            ),
            NodeWithScore(
                node=TextNode(text="Prayer is one of the five pillars of Islam."),
                score=0.7
            ),
            NodeWithScore(
                node=TextNode(text="Fasting in Ramadan is obligatory for adult Muslims."),
                score=0.75
            ),
        ]
        
        logger.info("Testing reranking...")
        logger.info(f"Query: {query}")
        logger.info(f"Nodes: {len(nodes)}")
        
        start_time = time.time()
        ranked_nodes = self.rerank(nodes, query)
        elapsed_time = time.time() - start_time
        
        result = {
            "success": True,
            "backend": self.llm_backend,
            "llm_model": self.llm_model,
            "query": query,
            "num_nodes": len(nodes),
            "reranking_time": round(elapsed_time, 3),
            "top_nodes_returned": len(ranked_nodes),
            "top_node_score": ranked_nodes[0].score if ranked_nodes else 0,
            "top_node_preview": ranked_nodes[0].node.get_content()[:100] if ranked_nodes else ""
        }
        
        logger.info("Test successful!")
        logger.info(f"  Reranking time: {elapsed_time:.3f}s")
        logger.info(f"  Top node: {result['top_node_preview']}")
        
        return result


def main():
    """Main function to test the reranker service."""
    print(f"Testing LlamaIndex LLMRerank with {config.LLM_BACKEND}")
    
    # Initialize service
    service = RerankerService(top_n=2)
    
    # Check availability
    if not service.check_availability():
        print("Reranker not available")
        return
    
    print("Reranker is available\n")
    
    # Run test
    result = service.test_reranking()
    print("Test result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("Reranker test complete!")


if __name__ == "__main__":
    main()

