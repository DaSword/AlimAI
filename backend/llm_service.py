"""
LLM Service - Simplified wrapper for Ollama LLM via LlamaIndex.

"""

import time
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

from backend.config import config
from backend.utils import setup_logging

logger = setup_logging("llm_service")


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # 'system', 'user', or 'assistant'
    content: str


@dataclass
class ChatResponse:
    """Represents a chat completion response."""
    content: str
    model: str
    generation_time: float
    total_tokens: Optional[int] = None


class LLMService:
    """
    Simplified service for direct LLM completions via LlamaIndex Ollama.
    
    Use this when you need:
    - Direct LLM calls without RAG
    - Manual conversation management
    - Simple completions or chat
    
    Use Chat Engines (see example_chat.py) when you need:
    - RAG with vector store retrieval
    - Automatic conversation history
    - Context-aware responses
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLMService.
        
        Args:
            model: Ollama model name (defaults to config)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt for conversations
        """
        self.model_name = model or config.OLLAMA_CHAT_MODEL
        self.temperature = temperature
        self.max_tokens = config.OLLAMA_MAX_TOKENS
        self.system_prompt = system_prompt
        self.ollama_llm = None
        
        logger.info(f"Initialized LLMService")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Max Tokens: {self.max_tokens or 'unlimited'}")
    
    def get_llm(self):
        """
        Get the underlying Ollama LLM instance.
        
        Returns:
            Ollama LLM instance (lazily loaded)
        """
        if self.ollama_llm is None:
            self._load_ollama_llm()
        return self.ollama_llm
        
    def _load_ollama_llm(self) -> bool:
        """
        Load the Ollama LLM via LlamaIndex.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            
            logger.info(f"📥 Loading Ollama LLM: {self.model_name}")
            
            # Build kwargs for Ollama
            kwargs = {
                "model": self.model_name,
                "base_url": config.OLLAMA_URL,
                "temperature": self.temperature,
                "request_timeout": config.OLLAMA_REQUEST_TIMEOUT,
                "context_window": config.OLLAMA_MAX_TOKENS,
            }
            
            # Add max_tokens if specified
            if self.max_tokens:
                kwargs["context_window"] = self.max_tokens
            
            self.ollama_llm = Ollama(**kwargs)
            
            logger.info(f"✅ Ollama LLM loaded successfully")
            return True
            
        except ImportError:
            logger.error("✗ llama-index-llms-ollama not installed")
            logger.error("  Install with: pip install llama-index-llms-ollama")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to load Ollama LLM: {str(e)}")
            logger.error(f"  Make sure Ollama is running at {config.OLLAMA_URL}")
            logger.error(f"  And model '{self.model_name}' is available")
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if the LLM model can be loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        if self.ollama_llm is None:
            try:
                self._load_ollama_llm()
                return True
            except Exception as e:
                logger.error(f"Error checking model availability: {str(e)}")
                return False
        return True
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None
    ) -> Optional[ChatResponse]:
        """
        Generate a chat completion.
        
        Note: For RAG-based chat with automatic retrieval and conversation
        management, use Chat Engines instead (see example_chat.py).
        
        Args:
            message: User message
            system_prompt: System prompt (overrides default)
            conversation_history: Previous messages in conversation
            
        Returns:
            ChatResponse or None if generation fails
        """
        if not message or not message.strip():
            logger.warning("Empty message provided for chat")
            return None
        
        # Ensure model is loaded
        if not self.check_model_availability():
            logger.error("LLM model not available")
            return None
        
        try:
            
            # Build messages list
            messages = []
            
            # Add system prompt
            system = system_prompt or self.system_prompt
            if system:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system))
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    if msg.role == "user":
                        role = MessageRole.USER
                    elif msg.role == "assistant":
                        role = MessageRole.ASSISTANT
                    elif msg.role == "system":
                        role = MessageRole.SYSTEM
                    else:
                        role = MessageRole.USER
                    messages.append(ChatMessage(role=role, content=msg.content))
            
            # Add current user message
            messages.append(ChatMessage(role=MessageRole.USER, content=message.strip()))
            
            # Generate completion
            start_time = time.time()
            response = self.ollama_llm.chat(messages)
            elapsed_time = time.time() - start_time
            
            # Build response object
            chat_response = ChatResponse(
                content=response.message.content,
                model=self.model_name,
                generation_time=elapsed_time
            )
            
            logger.info(f"✓ Generated response in {elapsed_time:.2f}s")
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            logger.error(f"  This might be due to insufficient memory for the model")
            logger.error(f"  Consider using a smaller model or increase available memory")
            return None
    
    def stream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Message]] = None
    ) -> Iterator[str]:
        """
        Generate a streaming chat completion.
        
        Args:
            message: User message
            system_prompt: System prompt (overrides default)
            conversation_history: Previous messages in conversation
            
        Yields:
            Chunks of the response as they are generated
        """
        if not message or not message.strip():
            logger.warning("Empty message provided for streaming chat")
            return
        
        # Ensure model is loaded
        if not self.check_model_availability():
            logger.error("LLM model not available")
            return
        
        try:
            
            # Build messages list
            messages = []
            
            # Add system prompt
            system = system_prompt or self.system_prompt
            if system:
                messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system))
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history:
                    if msg.role == "user":
                        role = MessageRole.USER
                    elif msg.role == "assistant":
                        role = MessageRole.ASSISTANT
                    elif msg.role == "system":
                        role = MessageRole.SYSTEM
                    else:
                        role = MessageRole.USER
                    messages.append(ChatMessage(role=role, content=msg.content))
            
            # Add current user message
            messages.append(ChatMessage(role=MessageRole.USER, content=message.strip()))
            
            # Stream completion
            logger.info("Streaming chat response...")
            response_stream = self.ollama_llm.stream_chat(messages)
            
            for chunk in response_stream:
                if chunk.delta:
                    yield chunk.delta
            
        except Exception as e:
            logger.error(f"Error streaming chat completion: {str(e)}")
            yield f"Error: {str(e)}"
    
    def complete(self, prompt: str) -> Optional[str]:
        """
        Generate a simple text completion (non-chat format).
        
        Args:
            prompt: Text prompt
            
        Returns:
            Generated text or None if generation fails
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided for completion")
            return None
        
        # Ensure model is loaded
        if not self.check_model_availability():
            logger.error("LLM model not available")
            return None
        
        try:
            response = self.ollama_llm.complete(prompt.strip())
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return None
    
    def stream_complete(self, prompt: str) -> Iterator[str]:
        """
        Generate a streaming text completion (non-chat format).
        
        Args:
            prompt: Text prompt
            
        Yields:
            Chunks of the response as they are generated
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided for streaming completion")
            return
        
        # Ensure model is loaded
        if not self.check_model_availability():
            logger.error("LLM model not available")
            return
        
        try:
            response_stream = self.ollama_llm.stream_complete(prompt.strip())
            
            for chunk in response_stream:
                if chunk.delta:
                    yield chunk.delta
                    
        except Exception as e:
            logger.error(f"Error streaming completion: {str(e)}")
            yield f"Error: {str(e)}"


def main():
    """Main function to test the LLM service."""
    # Initialize service
    service = LLMService(temperature=0.7)
    
    # Print configuration
    service.print_info()
    
    # Check model availability
    if not service.check_model_availability():
        print("❌ Failed to load LLM model")
        return
    
    # Run test
    result = service.test_chat()
    print(f"\nTest result: {result}")
    
    # Test streaming
    print("\n" + "="*60)
    print("Testing streaming chat...")
    print("="*60)
    
    test_message = "Briefly explain the concept of Tawhid in Islam."
    print(f"Message: {test_message}\n")
    print("Response (streaming):")
    
    for chunk in service.stream_chat(test_message):
        print(chunk, end="", flush=True)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

