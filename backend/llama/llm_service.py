"""
LLM Service - Simplified wrapper for Ollama LLM via LlamaIndex.

"""

import time
from typing import List, Optional, Iterator
from dataclasses import dataclass

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

from backend.core.config import config
from backend.core.utils import setup_logging

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
    Simplified service for direct LLM completions via LlamaIndex.
    
    Supports multiple backends:
    - Llama.cpp (default, OpenAI-compatible)
    - Ollama
    - LM Studio (local server)
    
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
        llm_backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLMService.
        
        Args:
            llm_backend: LLM backend ('llamacpp', 'ollama' or 'lmstudio', defaults to config)
            model: LLM model name (defaults to config based on backend)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt for conversations
        """
        self.llm_backend = llm_backend or config.LLM_BACKEND
        
        # Set model name based on backend
        if self.llm_backend == "llamacpp":
            self.model_name = model or config.LLAMACPP_CHAT_MODEL
            self.max_tokens = max_tokens or config.LLAMACPP_MAX_TOKENS
        elif self.llm_backend == "lmstudio":
            self.model_name = model or config.LMSTUDIO_CHAT_MODEL
            self.max_tokens = max_tokens or config.LMSTUDIO_MAX_TOKENS
        else:  # ollama
            self.model_name = model or config.OLLAMA_CHAT_MODEL
            self.max_tokens = max_tokens or config.OLLAMA_MAX_TOKENS
        
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.llm = None
        
        logger.info("Initialized LLMService")
        logger.info(f"  Backend: {self.llm_backend}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Temperature: {self.temperature}")
        logger.info(f"  Max Tokens: {self.max_tokens or 'unlimited'}")
    
    def get_llm(self):
        """
        Get the underlying LLM instance.
        
        Returns:
            LLM instance (lazily loaded)
        """
        if self.llm is None:
            self._load_llm()
        return self.llm
        
    def _load_llm(self) -> bool:
        """
        Load the LLM via LlamaIndex based on configured backend.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if self.llm_backend == "llamacpp":
                from llama_index.llms.openai import OpenAI
                
                logger.info(f"Loading Llama.cpp LLM: {self.model_name}")
                logger.info(f"  Llama.cpp URL: {config.LLAMACPP_CHAT_URL}")
                
                # Build kwargs for Llama.cpp (OpenAI-compatible)
                # Use a valid OpenAI model name to bypass validation, but the actual model
                # used will be whatever is loaded in llama.cpp server
                kwargs = {
                    "model": "gpt-3.5-turbo",  # Dummy model name to bypass validation
                    "api_base": config.LLAMACPP_CHAT_URL,
                    "api_key": "llama-cpp",  # Dummy key for OpenAI-compatible API
                    "temperature": self.temperature,
                    "timeout": config.LLAMACPP_REQUEST_TIMEOUT,
                    "is_chat_model": True,  # Tell LlamaIndex this is a chat model
                }
                
                self.llm = OpenAI(**kwargs)
                logger.info("✓ Llama.cpp LLM loaded successfully")
                logger.info(f"  Note: Using llama.cpp model '{self.model_name}' via OpenAI-compatible API")
                return True
            
            elif self.llm_backend == "ollama":
                from llama_index.llms.ollama import Ollama
                
                logger.info(f"Loading Ollama LLM: {self.model_name}")
                
                # Build kwargs for Ollama
                kwargs = {
                    "model": self.model_name,
                    "base_url": config.OLLAMA_URL,
                    "temperature": self.temperature,
                    "request_timeout": config.OLLAMA_REQUEST_TIMEOUT,
                    "context_window": self.max_tokens or config.OLLAMA_MAX_TOKENS,
                }
                
                self.llm = Ollama(**kwargs)
                logger.info("✓ Ollama LLM loaded successfully")
                return True
            
            elif self.llm_backend == "lmstudio":
                from llama_index.llms.lmstudio import LMStudio
                
                logger.info(f"Loading LM Studio LLM: {self.model_name}")
                logger.info(f"  LM Studio Request Timeout: {config.LMSTUDIO_REQUEST_TIMEOUT}s")
                
                # Build kwargs for LM Studio
                kwargs = {
                    "model_name": self.model_name,
                    "base_url": config.LMSTUDIO_URL,
                    "temperature": self.temperature,
                    "request_timeout": config.LMSTUDIO_REQUEST_TIMEOUT,  # Use request_timeout, not timeout!
                    "timeout": config.LMSTUDIO_REQUEST_TIMEOUT,  # Set both for completeness
                }
                
                self.llm = LMStudio(**kwargs)
                logger.info("✓ LM Studio LLM loaded successfully")
                return True
            
            else:
                logger.error(f"Unknown LLM backend: {self.llm_backend}")
                return False
            
        except ImportError as e:
            logger.error(f"Required package not installed: {str(e)}")
            if self.llm_backend == "llamacpp":
                logger.error("  Install with: pip install llama-index-llms-openai")
            elif self.llm_backend == "ollama":
                logger.error("  Install with: pip install llama-index-llms-ollama")
            else:
                logger.error("  Install with: pip install llama-index-llms-lmstudio")
            return False
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            if self.llm_backend == "llamacpp":
                logger.error(f"  Make sure llama.cpp server is running at {config.LLAMACPP_CHAT_URL}")
            elif self.llm_backend == "ollama":
                logger.error(f"  Make sure Ollama is running at {config.OLLAMA_URL}")
            else:
                logger.error(f"  Make sure LM Studio is running at {config.LMSTUDIO_URL}")
            logger.error(f"  And model '{self.model_name}' is available")
            return False
    
    def check_model_availability(self) -> bool:
        """
        Check if the LLM model can be loaded.
        
        Returns:
            True if model is available, False otherwise
        """
        if self.llm is None:
            try:
                self._load_llm()
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
            response = self.llm.chat(messages)
            elapsed_time = time.time() - start_time
            
            # Build response object
            chat_response = ChatResponse(
                content=response.message.content,
                model=self.model_name,
                generation_time=elapsed_time
            )
            
            logger.info(f"Generated response in {elapsed_time:.2f}s")
            
            return chat_response
            
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            logger.error("  This might be due to insufficient memory for the model")
            logger.error("  Consider using a smaller model or increase available memory")
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
            response_stream = self.llm.stream_chat(messages)
            
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
            response = self.llm.complete(prompt.strip())
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
            response_stream = self.llm.stream_complete(prompt.strip())
            
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
    
    # Check model availability
    if not service.check_model_availability():
        print("Failed to load LLM model")
        return
    
    # Run test
    result = service.test_chat()
    print(f"Test result: {result}")
    
    # Test streaming
    print("Testing streaming chat...")
    
    test_message = "Briefly explain the concept of Tawhid in Islam."
    print(f"Message: {test_message}\n")
    print("Response (streaming):")
    
    for chunk in service.stream_chat(test_message):
        print(chunk, end="", flush=True)
    


if __name__ == "__main__":
    main()

