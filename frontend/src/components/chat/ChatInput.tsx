import { useRef, useEffect } from "react";
import { Loader2, ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onCancel?: () => void;
  isLoading: boolean;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  onCancel,
  isLoading,
  disabled = false,
  placeholder = "Ask me anything about Islam..."
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading) {
        onSend();
      }
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [value]);

  return (
    <div className="border-t border-border bg-background/95 backdrop-blur-sm">
      <div className="max-w-3xl mx-auto px-4 py-4">
        <div className="relative bg-muted/50 rounded-2xl border border-border focus-within:border-primary focus-within:ring-2 focus-within:ring-primary/20 transition-all shadow-sm">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="w-full bg-transparent px-4 py-3 pr-12 resize-none focus:outline-none text-sm max-h-[200px]"
            disabled={isLoading || disabled}
            rows={1}
            style={{ minHeight: '48px' }}
          />
          <Button
            onClick={isLoading ? onCancel : onSend}
            disabled={!isLoading && (!value.trim() || disabled)}
            size="sm"
            className="absolute right-2 bottom-2 h-8 w-8 p-0 rounded-lg islamic-gradient shadow-md hover:shadow-lg transition-all"
            title={isLoading ? "Stop generation" : "Send message"}
          >
            {isLoading ? (
              <Square className="w-3.5 h-3.5 fill-current animate-pulse" />
            ) : (
              <ArrowUp className="w-4 h-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2 text-center">
          Alim AI can make mistakes. Always verify important information with scholars.
        </p>
      </div>
    </div>
  );
}

