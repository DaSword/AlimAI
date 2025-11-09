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
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [chatThreads, setChatThreads] = useState<ChatThread[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isAdminModalOpen, setIsAdminModalOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<ThemeMode>('dark');
  const [fullName, setFullName] = useState("Alim Admin");
  const [email, setEmail] = useState("admin@alim.ai");
  const [isEditingName, setIsEditingName] = useState(false);
  const [isEditingEmail, setIsEditingEmail] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Create a new thread on mount
    initializeThread();
  }, []);

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingMessage]);

  useEffect(() => {
    // Load saved theme preference
    const savedTheme = localStorage.getItem('theme') as ThemeMode || 'dark';
    setThemeMode(savedTheme);
    applyTheme(savedTheme);
  }, []);

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
    // Save current thread if it has messages
    if (messages.length > 0 && threadId) {
      const firstUserMessage = messages.find(m => m.role === "user");
      const title = firstUserMessage?.content.slice(0, 50) || "New Chat";
      setChatThreads(prev => [{
        id: threadId,
        title,
        timestamp: new Date()
      }, ...prev]);
    }
    
    setMessages([]);
    setStreamingMessage("");
    await initializeThread();
  };

  const handleSendMessage = async () => {
    if (!input.trim() || !threadId || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStreamingMessage("");

    try {
      let accumulatedContent = "";
      let sources: any[] = [];

      // Convert current messages to ChatMessage format for conversation history
      // Include both user and assistant messages with their original roles
      const conversationHistory: ChatMessage[] = messages.map(msg => ({
        role: msg.role, // Preserves "user" or "assistant" 
        content: msg.content
      }));

      for await (const event of streamChatResponse(threadId, userMessage.content, conversationHistory)) {
        // Handle different event types from LangGraph
        
        // Custom streaming events (token-by-token from get_stream_writer)
        if (event.event === "custom") {
          if (event.data?.token) {
            // Append token to accumulated content
            accumulatedContent += event.data.token;
            setStreamingMessage(accumulatedContent);
          } else if (event.data?.response) {
            // Fallback: use full response if available
            accumulatedContent = event.data.response;
            setStreamingMessage(accumulatedContent);
          }
        }
        
        // State values (complete state after node execution)
        if (event.event === "values") {
          // Extract the assistant's response
          if (event.data?.messages) {
            const lastMessage = event.data.messages[event.data.messages.length - 1];
            if (lastMessage?.role === "assistant") {
              accumulatedContent = lastMessage.content;
              setStreamingMessage(accumulatedContent);
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

      setMessages((prev) => [...prev, assistantMessage]);
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
      setMessages((prev) => [...prev, errorMessage]);
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
