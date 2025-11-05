import { cn } from "@/lib/utils";
import { IslamicBookIcon } from "./IslamicDecorations";
import type { ChatMessage } from "@/api/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Message extends ChatMessage {
  id: string;
  timestamp: Date;
  sources?: Array<{
    book_title: string;
    reference: string;
    text_content: string;
  }>;
}

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-4 animate-fadeIn", isUser && "justify-end")}>
      {isUser ? (
        <div className="flex flex-col items-end gap-2 max-w-[80%]">
          <div className="bg-primary text-primary-foreground rounded-2xl px-4 py-3 shadow-sm">
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      ) : (
        <div className="flex gap-3 max-w-[90%]">
          <div className="w-8 h-8 rounded-full islamic-gradient flex items-center justify-center flex-shrink-0 mt-1 shadow-md">
            <IslamicBookIcon className="w-5 h-5 text-white" />
          </div>

          <div className="flex flex-col gap-2 flex-1">
            <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-2 prose-p:leading-relaxed prose-headings:mt-4 prose-headings:mb-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-1 prose-strong:text-primary dark:prose-strong:text-primary prose-blockquote:border-l-primary prose-blockquote:bg-muted/30 prose-blockquote:py-1">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {isStreaming && (
                <span className="inline-block w-1.5 h-4 ml-1 bg-primary animate-pulse" />
              )}
            </div>

            {message.sources && message.sources.length > 0 && (
              <div className="mt-2 space-y-2">
                <p className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                  <span className="gold-accent">✦</span> Sources
                </p>
                <div className="space-y-2">
                  {message.sources.map((source, idx) => (
                    <div
                      key={idx}
                      className="text-xs bg-muted/50 rounded-lg p-3 space-y-1 islamic-card-border"
                    >
                      <p className="font-medium text-primary">
                        {source.book_title} - {source.reference}
                      </p>
                      <p className="text-muted-foreground line-clamp-2">
                        {source.text_content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

