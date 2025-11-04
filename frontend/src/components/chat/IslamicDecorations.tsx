// Islamic Book of Knowledge icon
export const IslamicBookIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Book base */}
    <path d="M25 20 L75 20 L75 85 L25 85 Z" fill="currentColor" opacity="0.3" />
    
    {/* Book spine shadow */}
    <path d="M25 20 L30 25 L30 90 L25 85 Z" fill="currentColor" opacity="0.5" />
    
    {/* Book pages */}
    <rect x="30" y="25" width="40" height="60" fill="currentColor" opacity="0.2" />
    
    {/* Decorative border */}
    <rect x="33" y="28" width="34" height="54" stroke="currentColor" strokeWidth="1" opacity="0.4" fill="none" />
    <rect x="35" y="30" width="30" height="50" stroke="currentColor" strokeWidth="0.5" opacity="0.3" fill="none" />
    
    {/* Islamic geometric pattern on cover */}
    <g transform="translate(50, 55)">
      {/* 8-pointed star */}
      <path d="M0 -12 L3 -5 L10 -6 L5 -2 L7 5 L0 0 L-7 5 L-5 -2 L-10 -6 L-3 -5 Z" 
            fill="currentColor" opacity="0.6" />
      <circle cx="0" cy="0" r="3" fill="currentColor" opacity="0.7" />
    </g>
    
    {/* Decorative dots */}
    <circle cx="50" cy="35" r="1" fill="currentColor" opacity="0.5" />
    <circle cx="50" cy="75" r="1" fill="currentColor" opacity="0.5" />
    
    {/* Page lines suggesting text */}
    <line x1="38" y1="40" x2="62" y2="40" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
    <line x1="38" y1="45" x2="62" y2="45" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
    <line x1="38" y1="50" x2="58" y2="50" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
    <line x1="38" y1="65" x2="62" y2="65" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
    <line x1="38" y1="70" x2="60" y2="70" stroke="currentColor" strokeWidth="0.5" opacity="0.3" />
    
    {/* Book bookmark */}
    <path d="M55 20 L55 45 L52.5 42 L50 45 L50 20 Z" fill="currentColor" opacity="0.6" />
  </svg>
);

// Decorative corner ornament
export const CornerOrnament = ({ className, position }: { className?: string; position: 'tl' | 'tr' | 'bl' | 'br' }) => {
  const rotations = { tl: 0, tr: 90, bl: 270, br: 180 };
  return (
    <svg 
      className={className} 
      viewBox="0 0 60 60" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
      style={{ transform: `rotate(${rotations[position]}deg)` }}
    >
      <path d="M0 0 Q15 0 20 5 Q25 10 25 20 L25 25 Q25 15 30 10 Q35 5 45 5 L50 5" 
            stroke="currentColor" strokeWidth="0.8" opacity="0.3" fill="none" />
      <path d="M5 0 Q18 2 22 8 Q26 14 26 24" 
            stroke="currentColor" strokeWidth="0.6" opacity="0.25" fill="none" />
      <circle cx="20" cy="20" r="2" fill="currentColor" opacity="0.3" />
      <circle cx="30" cy="10" r="1.5" fill="currentColor" opacity="0.25" />
      <circle cx="10" cy="30" r="1.5" fill="currentColor" opacity="0.25" />
    </svg>
  );
};

// Decorative divider
export const IslamicDivider = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 200 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Main wave line */}
    <path 
      d="M0 20 Q25 10 40 20 T80 20 T120 20 T160 20 T200 20" 
      strokeWidth="1.5" 
      fill="none"
      className="dark:hidden"
      stroke="hsl(220 40% 25% / 0.7)"
    />
    <path 
      d="M0 20 Q25 10 40 20 T80 20 T120 20 T160 20 T200 20" 
      strokeWidth="1.5" 
      fill="none"
      className="hidden dark:block"
      stroke="hsl(45 100% 51% / 0.6)"
    />
    
    {/* Center ornament circles */}
    <circle cx="100" cy="20" r="6" className="dark:hidden" fill="hsl(160 84% 39% / 0.8)" />
    <circle cx="100" cy="20" r="6" className="hidden dark:block" fill="hsl(45 100% 51% / 0.7)" />
    <circle cx="100" cy="20" r="3" className="dark:hidden" fill="hsl(220 40% 25% / 0.9)" />
    <circle cx="100" cy="20" r="3" className="hidden dark:block" fill="hsl(160 84% 39% / 0.8)" />
    
    {/* Side circles */}
    <circle cx="50" cy="20" r="3" className="dark:hidden" fill="hsl(160 84% 39% / 0.7)" />
    <circle cx="50" cy="20" r="3" className="hidden dark:block" fill="hsl(45 100% 51% / 0.6)" />
    <circle cx="150" cy="20" r="3" className="dark:hidden" fill="hsl(160 84% 39% / 0.7)" />
    <circle cx="150" cy="20" r="3" className="hidden dark:block" fill="hsl(45 100% 51% / 0.6)" />
    
    {/* Decorative diamonds */}
    <path d="M82 20 L88 13 L94 20 L88 27 Z" className="dark:hidden" fill="hsl(173 80% 40% / 0.6)" />
    <path d="M82 20 L88 13 L94 20 L88 27 Z" className="hidden dark:block" fill="hsl(160 84% 39% / 0.5)" />
    <path d="M106 20 L112 13 L118 20 L112 27 Z" className="dark:hidden" fill="hsl(173 80% 40% / 0.6)" />
    <path d="M106 20 L112 13 L118 20 L112 27 Z" className="hidden dark:block" fill="hsl(160 84% 39% / 0.5)" />
    
    {/* Small accent dots */}
    <circle cx="25" cy="20" r="2" className="dark:hidden" fill="hsl(220 40% 25% / 0.5)" />
    <circle cx="25" cy="20" r="2" className="hidden dark:block" fill="hsl(45 100% 51% / 0.4)" />
    <circle cx="175" cy="20" r="2" className="dark:hidden" fill="hsl(220 40% 25% / 0.5)" />
    <circle cx="175" cy="20" r="2" className="hidden dark:block" fill="hsl(45 100% 51% / 0.4)" />
  </svg>
);

