import React, { useRef, useEffect } from "react";
import { Plus, PanelLeft, ChevronDown, Settings, ShieldCheck, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatThread {
  id: string;
  title: string;
  timestamp: Date;
}

interface ChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewConversation: () => void;
  chatThreads: ChatThread[];
  fullName: string;
  onOpenSettings: () => void;
  onOpenAdmin: () => void;
  onLogout?: () => void;
}

export function ChatSidebar({
  isOpen,
  onClose,
  onNewConversation,
  chatThreads,
  fullName,
  onOpenSettings,
  onOpenAdmin,
  onLogout
}: ChatSidebarProps) {
  const userMenuRef = useRef<HTMLDivElement>(null);
  const [isUserMenuOpen, setIsUserMenuOpen] = React.useState(false);

  useEffect(() => {
    // Close user menu when clicking outside
    const handleClickOutside = (event: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    };

    if (isUserMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isUserMenuOpen]);

  return (
    <div
      className={cn(
        "flex flex-col bg-card border-r border-border transition-all duration-300 overflow-hidden",
        isOpen ? "w-64" : "w-0"
      )}
    >
      <div className="flex items-center justify-between p-4 border-b border-border islamic-card-border min-w-64">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="w-9 h-9 rounded-lg bg-white/10 flex items-center justify-center shadow-md flex-shrink-0">
            <img src="/logo.png" alt="Alim AI" className="w-8 h-8 object-contain" />
          </div>
          <span className="font-semibold text-sm heading-elegant whitespace-nowrap">Alim AI</span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="h-8 w-8 p-0 flex-shrink-0"
        >
          <PanelLeft className="w-4 h-4" />
        </Button>
      </div>

      <div className="p-3 min-w-64">
        <Button
          onClick={onNewConversation}
          className="w-full justify-start gap-2 whitespace-nowrap"
          variant="outline"
        >
          <Plus className="w-4 h-4 flex-shrink-0" />
          New chat
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 min-w-64">
        <div className="text-xs text-muted-foreground mb-2 px-2 whitespace-nowrap">Recent</div>
        {chatThreads.map((thread) => (
          <button
            key={thread.id}
            className="w-full text-left px-3 py-2 rounded-lg hover:bg-muted/50 transition-colors text-sm mb-1 truncate overflow-hidden whitespace-nowrap text-ellipsis"
          >
            {thread.title}
          </button>
        ))}
      </div>

      <div className="p-3 border-t border-border relative min-w-64" ref={userMenuRef}>
        <Button 
          variant="ghost" 
          size="sm" 
          className="w-full justify-between gap-2 group"
          onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
        >
          <div className="flex items-center gap-2 overflow-hidden">
            <div className="w-7 h-7 rounded-full islamic-gradient flex items-center justify-center text-white text-xs font-semibold flex-shrink-0">
              {fullName.split(' ').map(n => n[0]).join('')}
            </div>
            <span className="text-sm whitespace-nowrap overflow-hidden text-ellipsis">{fullName}</span>
          </div>
          <ChevronDown className={cn("w-4 h-4 transition-transform flex-shrink-0", isUserMenuOpen && "rotate-180")} />
        </Button>
        
        {/* User Menu Popup */}
        {isUserMenuOpen && (
          <div className="absolute bottom-full left-3 right-3 mb-2 bg-card border border-border rounded-lg shadow-lg overflow-hidden animate-fadeIn">
            <button
              onClick={() => {
                onOpenSettings();
                setIsUserMenuOpen(false);
              }}
              className="w-full px-4 py-2.5 text-left text-sm hover:bg-muted/50 transition-colors flex items-center gap-2"
            >
              <Settings className="w-4 h-4" />
              Settings
            </button>
            <button
              onClick={() => {
                onOpenAdmin();
                setIsUserMenuOpen(false);
              }}
              className="w-full px-4 py-2.5 text-left text-sm hover:bg-muted/50 transition-colors flex items-center gap-2"
            >
              <ShieldCheck className="w-4 h-4" />
              Admin
            </button>
            <button
              onClick={() => {
                setIsUserMenuOpen(false);
                onLogout?.();
              }}
              className="w-full px-4 py-2.5 text-left text-sm hover:bg-muted/50 transition-colors flex items-center gap-2 text-red-500"
            >
              <LogOut className="w-4 h-4" />
              Log out
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

