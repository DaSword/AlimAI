import { useState, useEffect, useRef } from "react";
import { PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { createThread, streamChatResponse } from "@/api/client";
import type { ChatMessage } from "@/api/client";
import {
  ChatSidebar,
  SettingsModal,
  AdminModal,
  WelcomeScreen,
  ChatInput,
  MessageBubble,
  type ThemeMode
} from "@/components/chat";

interface Message extends ChatMessage {
  id: string;
  timestamp: Date;
  sources?: Array<{
    book_title: string;
    reference: string;
    text_content: string;
  }>;
}

interface ChatThread {
  id: string;
  title: string;
  timestamp: Date;
  messages: Message[];
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [chatThreads, setChatThreads] = useState<ChatThread[]>([]);
  const [currentThreadIndex, setCurrentThreadIndex] = useState<number | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isAdminModalOpen, setIsAdminModalOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<ThemeMode>('dark');
  const [fullName, setFullName] = useState("Alim Admin");
  const [email, setEmail] = useState("admin@alim.ai");
  const [isEditingName, setIsEditingName] = useState(false);
  const [isEditingEmail, setIsEditingEmail] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const currentThreadIndexRef = useRef<number | null>(null);

  useEffect(() => {
    // Load chat history from localStorage
    const savedThreads = localStorage.getItem('chatThreads');
    if (savedThreads) {
      try {
        const parsed = JSON.parse(savedThreads);
        // Convert timestamp strings back to Date objects
        const threads = parsed.map((thread: any) => ({
          ...thread,
          timestamp: new Date(thread.timestamp),
          messages: thread.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setChatThreads(threads);
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
    
    // Create a new thread on mount
    initializeThread();
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingMessage]);

  useEffect(() => {
    // Keep ref in sync with state
    currentThreadIndexRef.current = currentThreadIndex;
  }, [currentThreadIndex]);

  useEffect(() => {
    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme') as ThemeMode || 'dark';
    setThemeMode(savedTheme);
    applyTheme(savedTheme);
  }, []);

  // Save chatThreads to localStorage whenever it changes
  useEffect(() => {
    if (chatThreads.length > 0) {
      localStorage.setItem('chatThreads', JSON.stringify(chatThreads));
    }
  }, [chatThreads]);

  const applyTheme = (mode: ThemeMode) => {
    const root = document.documentElement;
    
    if (mode === 'light') {
      root.classList.remove('dark');
    } else if (mode === 'dark') {
      root.classList.add('dark');
    } else {
      // System preference
      const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      if (systemPrefersDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
    
    localStorage.setItem('theme', mode);
  };

  const handleThemeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
    applyTheme(mode);
  };

  const initializeThread = async () => {
    try {
      const newThreadId = await createThread();
      setThreadId(newThreadId);
    } catch (error) {
      console.error("Failed to create thread:", error);
    }
  };

  const handleNewConversation = async () => {
    // The current conversation is already auto-saved, so we just need to reset
    setMessages([]);
    setStreamingMessage("");
    setCurrentThreadIndex(null);
    await initializeThread();
  };

  const loadChat = (index: number) => {
    const thread = chatThreads[index];
    if (thread) {
      setThreadId(thread.id);
      setMessages(thread.messages);
      setCurrentThreadIndex(index);
      setStreamingMessage(""); // Clear any streaming from other threads
      setIsLoading(false); // Stop loading state if switching during a stream
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim() || !threadId || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput("");
    setIsLoading(true);
    setStreamingMessage("");

    // Capture the thread index and ID at the START of the message
    let messageThreadIndex = currentThreadIndex;
    const messageThreadId = threadId;

    // Create/save thread immediately on first user message
    if (currentThreadIndex === null && messages.length === 0) {
      const title = userMessage.content.slice(0, 50) || "New Chat";
      const newThread: ChatThread = {
        id: threadId,
        title,
        timestamp: new Date(),
        messages: updatedMessages
      };
      
      setChatThreads(prev => [newThread, ...prev]);
      setCurrentThreadIndex(0); // Set to the newly created thread
      messageThreadIndex = 0; // Capture the new index
    } else if (currentThreadIndex !== null) {
      // Update existing thread with the new user message immediately
      setChatThreads(prev => {
        const updated = [...prev];
        if (updated[currentThreadIndex]) {
          updated[currentThreadIndex] = {
            ...updated[currentThreadIndex],
            messages: updatedMessages
          };
        }
        return updated;
      });
    }

    try {
      let accumulatedContent = "";
      let sources: any[] = [];

      // Convert current messages to ChatMessage format for conversation history
      // Include both user and assistant messages with their original roles
      const conversationHistory: ChatMessage[] = messages.map(msg => ({
        role: msg.role, // Preserves "user" or "assistant" 
        content: msg.content
      }));

      for await (const event of streamChatResponse(messageThreadId, userMessage.content, conversationHistory)) {
        // Handle different event types from LangGraph
        
        // Custom streaming events (token-by-token from get_stream_writer)
        if (event.event === "custom") {
          if (event.data?.token) {
            // Append token to accumulated content
            accumulatedContent += event.data.token;
            // Only update streaming display if still viewing the same thread (use ref for real-time value)
            if (currentThreadIndexRef.current === messageThreadIndex) {
              setStreamingMessage(accumulatedContent);
            }
          } else if (event.data?.response) {
            // Fallback: use full response if available
            accumulatedContent = event.data.response;
            if (currentThreadIndexRef.current === messageThreadIndex) {
              setStreamingMessage(accumulatedContent);
            }
          }
        }
        
        // State values (complete state after node execution)
        if (event.event === "values") {
          // Extract the assistant's response
          if (event.data?.messages) {
            const lastMessage = event.data.messages[event.data.messages.length - 1];
            if (lastMessage?.role === "assistant") {
              accumulatedContent = lastMessage.content;
              if (currentThreadIndexRef.current === messageThreadIndex) {
                setStreamingMessage(accumulatedContent);
              }
            }
          }

          // Extract sources if available
          if (event.data?.sources) {
            sources = event.data.sources;
          }
        }
      }

      // Add the complete message
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: accumulatedContent,
        timestamp: new Date(),
        sources: sources.length > 0 ? sources : undefined,
      };

      // Save to the correct thread, even if user switched views
      if (messageThreadIndex !== null) {
        setChatThreads(prev => {
          const updated = [...prev];
          if (updated[messageThreadIndex]) {
            updated[messageThreadIndex] = {
              ...updated[messageThreadIndex],
              messages: [...updated[messageThreadIndex].messages, assistantMessage]
            };
          }
          return updated;
        });
      }

      // Only update visible messages if still viewing the same thread (use ref for real-time value)
      if (currentThreadIndexRef.current === messageThreadIndex) {
        setMessages((prev) => [...prev, assistantMessage]);
      }
      setStreamingMessage("");
    } catch (error: any) {
      console.error("Error sending message:", error);
      
      // Provide more specific error message
      let errorContent = "Sorry, I encountered an error. Please try again.";
      
      if (error?.message?.includes("timeout")) {
        errorContent = error.message;
      } else if (error?.message) {
        errorContent = `Error: ${error.message}`;
      }
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: errorContent,
        timestamp: new Date(),
      };

      // Save error to the correct thread
      if (messageThreadIndex !== null) {
        setChatThreads(prev => {
          const updated = [...prev];
          if (updated[messageThreadIndex]) {
            updated[messageThreadIndex] = {
              ...updated[messageThreadIndex],
              messages: [...updated[messageThreadIndex].messages, errorMessage]
            };
          }
          return updated;
        });
      }

      // Only update visible messages if still viewing the same thread (use ref for real-time value)
      if (currentThreadIndexRef.current === messageThreadIndex) {
        setMessages((prev) => [...prev, errorMessage]);
      }
      setStreamingMessage("");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-full islamic-pattern">
      {/* Sidebar */}
      <ChatSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        onNewConversation={handleNewConversation}
        chatThreads={chatThreads}
        currentThreadIndex={currentThreadIndex}
        onLoadChat={loadChat}
        fullName={fullName}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenAdmin={() => setIsAdminModalOpen(true)}
        onLogout={() => {
          // Log out logic will go here
        }}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-full">
        {/* Header */}
        {!isSidebarOpen && (
          <div className="flex items-center gap-3 p-4 border-b border-border">
            <div className="w-9 h-9 rounded-lg bg-white/10 flex items-center justify-center shadow-md">
              <img src="/logo.png" alt="Alim AI" className="w-8 h-8 object-contain" />
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsSidebarOpen(true)}
              className="h-8 w-8 p-0"
            >
              <PanelLeft className="w-4 h-4" />
            </Button>
          </div>
        )}

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {messages.length === 0 && !streamingMessage ? (
            <WelcomeScreen onPromptClick={setInput} />
          ) : (
            <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
              {messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
              ))}

              {streamingMessage && (
                <MessageBubble
                  message={{
                    id: "streaming",
                    role: "assistant",
                    content: streamingMessage,
                    timestamp: new Date(),
                  }}
                  isStreaming
                />
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <ChatInput
          value={input}
          onChange={setInput}
          onSend={handleSendMessage}
          isLoading={isLoading}
          disabled={!threadId}
        />
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        fullName={fullName}
        email={email}
        themeMode={themeMode}
        isEditingName={isEditingName}
        isEditingEmail={isEditingEmail}
        onFullNameChange={setFullName}
        onEmailChange={setEmail}
        onThemeChange={handleThemeChange}
        onSetEditingName={setIsEditingName}
        onSetEditingEmail={setIsEditingEmail}
      />

      {/* Admin Modal */}
      <AdminModal
        isOpen={isAdminModalOpen}
        onClose={() => setIsAdminModalOpen(false)}
      />
    </div>
  );
}
