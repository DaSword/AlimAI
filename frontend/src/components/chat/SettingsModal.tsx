import { X, Sun, Moon, Monitor, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export type ThemeMode = 'light' | 'dark' | 'system';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  fullName: string;
  email: string;
  themeMode: ThemeMode;
  isEditingName: boolean;
  isEditingEmail: boolean;
  onFullNameChange: (name: string) => void;
  onEmailChange: (email: string) => void;
  onThemeChange: (mode: ThemeMode) => void;
  onSetEditingName: (editing: boolean) => void;
  onSetEditingEmail: (editing: boolean) => void;
}

function ThemeCard({
  icon,
  label,
  description,
  isSelected,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  description: string;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative p-4 rounded-lg border-2 transition-all text-left",
        isSelected
          ? "border-primary bg-primary/5 shadow-md"
          : "border-border hover:border-primary/50 bg-card"
      )}
    >
      <div className="flex flex-col items-center gap-3 text-center">
        <div className={cn(
          "w-12 h-12 rounded-full flex items-center justify-center transition-all",
          isSelected ? "islamic-gradient text-white shadow-lg" : "bg-muted text-muted-foreground"
        )}>
          {icon}
        </div>
        <div>
          <div className="font-medium text-sm">{label}</div>
          <div className="text-xs text-muted-foreground mt-1">{description}</div>
        </div>
      </div>
      {isSelected && (
        <div className="absolute top-2 right-2 w-2 h-2 rounded-full bg-primary" />
      )}
    </button>
  );
}

export function SettingsModal({
  isOpen,
  onClose,
  fullName,
  email,
  themeMode,
  isEditingName,
  isEditingEmail,
  onFullNameChange,
  onEmailChange,
  onThemeChange,
  onSetEditingName,
  onSetEditingEmail
}: SettingsModalProps) {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          onClose();
        }
      }}
    >
      <div className="bg-card rounded-2xl shadow-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden animate-fadeIn islamic-card-border">
        {/* Modal Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className="text-2xl font-semibold heading-elegant">Settings</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Modal Content */}
        <div className="overflow-y-auto max-h-[calc(85vh-80px)] custom-scrollbar">
          <div className="p-6 space-y-8">
            {/* General Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 heading-elegant">General</h3>
              <Card className="p-6 islamic-card-border">
                <div className="space-y-6">
                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Full name
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        placeholder="Your name"
                        value={fullName}
                        onChange={(e) => onFullNameChange(e.target.value)}
                        onBlur={() => onSetEditingName(false)}
                        readOnly={!isEditingName}
                        autoFocus={isEditingName}
                        className={cn(
                          "w-full px-3 py-2 pr-10 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all",
                          !isEditingName && "cursor-default text-muted-foreground"
                        )}
                      />
                      {!isEditingName && (
                        <button
                          onClick={() => onSetEditingName(true)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>

                  <div>
                    <label className="text-sm font-medium mb-2 block">
                      Email
                    </label>
                    <div className="relative">
                      <input
                        type="email"
                        placeholder="your.email@example.com"
                        value={email}
                        onChange={(e) => onEmailChange(e.target.value)}
                        onBlur={() => onSetEditingEmail(false)}
                        readOnly={!isEditingEmail}
                        autoFocus={isEditingEmail}
                        className={cn(
                          "w-full px-3 py-2 pr-10 bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all",
                          !isEditingEmail && "cursor-default text-muted-foreground"
                        )}
                      />
                      {!isEditingEmail && (
                        <button
                          onClick={() => onSetEditingEmail(true)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Appearance Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 heading-elegant">Appearance</h3>
              <Card className="p-6 islamic-card-border">
                <div>
                  <h4 className="font-medium mb-4">Theme</h4>
                  <div className="grid grid-cols-3 gap-4">
                    <ThemeCard
                      icon={<Sun className="w-5 h-5" />}
                      label="Light"
                      description="Bright and clear"
                      isSelected={themeMode === 'light'}
                      onClick={() => onThemeChange('light')}
                    />
                    <ThemeCard
                      icon={<Monitor className="w-5 h-5" />}
                      label="System"
                      description="Match system"
                      isSelected={themeMode === 'system'}
                      onClick={() => onThemeChange('system')}
                    />
                    <ThemeCard
                      icon={<Moon className="w-5 h-5" />}
                      label="Dark"
                      description="Easy on the eyes"
                      isSelected={themeMode === 'dark'}
                      onClick={() => onThemeChange('dark')}
                    />
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

