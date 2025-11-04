import { CornerOrnament, IslamicDivider } from "./IslamicDecorations";

interface WelcomeScreenProps {
  onPromptClick: (prompt: string) => void;
}

export function WelcomeScreen({ onPromptClick }: WelcomeScreenProps) {
  const quickActions = [
    { emoji: "📖", label: "Quran", prompt: "Tell me about Surah Al-Fatiha" },
    { emoji: "📿", label: "Hadith", prompt: "Share an authentic Hadith about kindness" },
    { emoji: "🕌", label: "Prayer", prompt: "How do I perform Wudu?" },
    { emoji: "🧠", label: "Fiqh", prompt: "What are the conditions of Zakat?" },
  ];

  return (
    <div className="h-full flex flex-col items-center justify-center px-4 py-12 relative">
      {/* Decorative Corner Ornaments */}
      <CornerOrnament className="absolute top-8 left-8 w-16 h-16 text-primary/20 hidden md:block animate-fadeIn" position="tl" />
      <CornerOrnament className="absolute top-8 right-8 w-16 h-16 text-primary/20 hidden md:block animate-fadeIn" position="tr" />
      <CornerOrnament className="absolute bottom-24 left-8 w-16 h-16 text-primary/20 hidden md:block animate-fadeIn" position="bl" />
      <CornerOrnament className="absolute bottom-24 right-8 w-16 h-16 text-primary/20 hidden md:block animate-fadeIn" position="br" />
      
      {/* Main Content Container */}
      <div className="max-w-3xl w-full space-y-8 animate-fadeIn">
        {/* Bismillah Section */}
        <div className="text-center space-y-4 relative">
          {/* Top decorative divider */}
          <IslamicDivider className="w-48 h-6 mx-auto mb-6 text-primary/30" />
          
          <div className="relative inline-block px-4">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent blur-xl"></div>
            <p className="relative arabic text-5xl md:text-6xl font-bold" 
               style={{ 
                 fontFamily: "'Scheherazade New', 'Amiri', serif",
                 background: 'linear-gradient(135deg, hsl(var(--islamic-gold)), hsl(var(--islamic-emerald)))',
                 WebkitBackgroundClip: 'text',
                 WebkitTextFillColor: 'transparent',
                 backgroundClip: 'text',
                 filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
                 lineHeight: '1.5',
                 paddingBottom: '1rem',
                 paddingTop: '0.75rem'
               }}>
              بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ
            </p>
          </div>
          
          <p className="text-sm text-muted-foreground font-medium tracking-wide">
            In the name of Allah, the Most Gracious, the Most Merciful
          </p>
          
          {/* Bottom decorative divider */}
          <IslamicDivider className="w-48 h-6 mx-auto mt-6 text-primary/30" />
        </div>
        
        {/* Icon with Glow Effect */}
        <div className="flex justify-center relative">
          <div className="relative">
            {/* Outer glow rings */}
            <div className="absolute inset-0 rounded-full bg-primary/10 blur-2xl animate-pulse" 
                 style={{ animationDuration: '3s' }}></div>
            <div className="absolute inset-0 rounded-full bg-accent/10 blur-xl animate-pulse" 
                 style={{ animationDuration: '2s', animationDelay: '0.5s' }}></div>
            
            {/* Main icon container */}
            <div className="relative w-40 h-40 rounded-full bg-white/10 backdrop-blur-sm flex items-center justify-center shadow-2xl transform hover:scale-105 transition-transform duration-300 border-4 border-primary/20">
              <img src="/logo.png" alt="Alim AI" className="w-32 h-32 object-contain drop-shadow-2xl" />
            </div>
          </div>
        </div>
        
        {/* Greeting Section */}
        <div className="text-center space-y-3">
          <p className="text-muted-foreground text-center max-w-xl mx-auto leading-relaxed pt-4">
            Ask me anything about Islam, the Quran, Hadith, Fiqh, and more. All answers are grounded in authentic Islamic sources.
          </p>
        </div>
        
        {/* Quick Action Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-6 max-w-2xl mx-auto">
          {quickActions.map((item) => (
            <button
              key={item.label}
              onClick={() => onPromptClick(item.prompt)}
              className="group relative p-4 rounded-xl border border-border bg-card hover:bg-muted/50 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity rounded-xl"></div>
              <div className="relative text-center space-y-2">
                <div className="text-3xl">{item.emoji}</div>
                <div className="text-sm font-medium text-foreground">{item.label}</div>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

