import { useMemo } from 'react';
import type { TwinState } from '@/data/demo';
import { fieldPlants } from '@/data/demo';

type FieldTwinProps = {
  state: TwinState;
};

const stateCopy: Record<TwinState, { title: string; subtitle: string; date: string; tint: string }> = {
  current: { title: 'Current state', subtitle: 'Physics-based simulation', date: 'DAY 42 · 18 JUN 2024', tint: '#728f71' },
  assimilated: { title: 'Assimilated state', subtitle: 'Observation-adjusted estimate', date: 'DAY 42 · UPDATED', tint: '#8fa382' },
  forecast: { title: 'Forecast state', subtitle: 'Projected growth trajectory', date: 'DAY 42 → 63', tint: '#3b82f6' },
};

export function FieldTwin({ state }: FieldTwinProps) {
  const copy = stateCopy[state];
  const shift = state === 'forecast' ? 2 : state === 'assimilated' ? -1 : 0;
  const highlighted = useMemo(() => state !== 'current', [state]);

  return (
    <div className="relative overflow-hidden rounded-lg border border-[#dfe3dc] bg-[#f4f0e5]">
      <div className="absolute inset-x-0 top-0 z-10 flex items-center justify-between border-b border-[#d6d9ce] bg-white/85 px-4 py-3 backdrop-blur-sm sm:px-6">
        <div>
          <p className="label-caps text-[#1b3022]">{copy.title}</p>
          <p className="mt-1 text-[11px] text-[#69736c]">{copy.subtitle}</p>
        </div>
        <p className="mono text-[10px] tracking-[.1em] text-[#69736c]">{copy.date}</p>
      </div>
      <svg className="block aspect-[1.62] w-full min-h-[310px]" viewBox="0 0 760 470" role="img" aria-label={`${copy.title} visualization of the wheat field`}>
        <defs>
          <linearGradient id="field-depth" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#dfe8d3" />
            <stop offset=".62" stopColor="#bdcaad" />
            <stop offset="1" stopColor="#a5b18f" />
          </linearGradient>
          <pattern id="survey-grid" width="35" height="29" patternUnits="userSpaceOnUse">
            <path d="M 35 0 L 0 0 0 29" fill="none" stroke="#ffffff" strokeOpacity=".15" strokeWidth="1" />
          </pattern>
          <filter id="field-shadow" x="-20%" y="-20%" width="140%" height="160%">
            <feGaussianBlur stdDeviation="11" />
          </filter>
        </defs>
        <rect width="760" height="470" fill="#f4f0e5" />
        <path d="M0 385 C140 350 230 399 350 369 C480 337 602 385 760 347 L760 470 L0 470 Z" fill="#e8e3d5" />
        <ellipse cx="390" cy="274" rx="315" ry="153" fill="#64745c" opacity=".13" filter="url(#field-shadow)" />
        <g transform={`translate(${shift}, ${state === 'forecast' ? -3 : 0})`} className="transition-transform duration-700">
          <path d="M74 62 L671 54 L710 364 L92 378 Z" fill="url(#field-depth)" stroke="#536b54" strokeWidth="2" />
          <clipPath id="field-clip">
            <path d="M74 62 L671 54 L710 364 L92 378 Z" />
          </clipPath>
          <g clipPath="url(#field-clip)">
            <path d="M74 62 L671 54 L710 364 L92 378 Z" fill="url(#survey-grid)" opacity=".42" />
            <path d="M93 105 C190 82 262 115 355 94 C465 70 565 112 685 91 L690 159 C564 176 481 140 376 157 C272 175 175 142 86 166 Z" fill="#e7eddb" opacity=".34" />
            <path d="M96 246 C206 221 292 258 393 232 C492 208 589 248 701 225 L706 308 C592 328 505 285 401 311 C292 338 194 294 96 321 Z" fill="#758764" opacity=".12" />
            <path d="M92 378 L710 364" stroke="#6d7e64" strokeWidth="2" opacity=".6" />
            <path d="M74 62 L92 378 M671 54 L710 364" stroke="#94a884" strokeWidth="1" strokeDasharray="4 6" opacity=".8" />
            {Array.from({ length: 10 }, (_, index) => {
              const y = 68 + index * 31;
              return <path key={`furrow-${index}`} d={`M82 ${y} L686 ${y - 5}`} stroke="#718364" strokeWidth="1.5" opacity=".45" />;
            })}
            <g className="transition-opacity duration-500" opacity={state === 'forecast' ? 0.84 : 1}>
              {fieldPlants.map((plant) => {
                const plantOpacity = state === 'forecast' ? Math.min(1, plant.opacity + 0.13) : highlighted ? Math.min(1, plant.opacity + 0.04) : plant.opacity;
                return (
                  <g key={`${plant.row}-${plant.column}`} transform={`translate(${plant.x} ${plant.y}) rotate(${plant.lean})`} opacity={plantOpacity}>
                    <path d={`M0 10 Q-1 ${plant.height * .55} 0 -${plant.height}`} fill="none" stroke={plant.row % 3 === 0 ? '#3f6945' : '#577c50'} strokeWidth={3} strokeLinecap="round" />
                    <path d={`M0 2 Q${-7 - plant.height / 3} -1 -7 ${plant.height / 3}`} fill="none" stroke="#789769" strokeWidth="2" strokeLinecap="round" />
                    <path d={`M0 -${plant.height * .3} Q${7 + plant.height / 3} -${plant.height * .1} 8 -${plant.height * .48}`} fill="none" stroke="#789769" strokeWidth="2" strokeLinecap="round" />
                    {plant.column % 4 === 0 && <circle cx="0" cy={-plant.height - 3} r="2.4" fill="#c49142" opacity=".7" />}
                  </g>
                );
              })}
            </g>
            <path d="M113 130 Q388 105 669 128" fill="none" stroke="#d7dfc8" strokeWidth="1" opacity=".8" />
            <path d="M105 232 Q405 206 687 223" fill="none" stroke="#d7dfc8" strokeWidth="1" opacity=".55" />
            <path d="M117 319 Q420 291 699 309" fill="none" stroke="#d7dfc8" strokeWidth="1" opacity=".45" />
          </g>
          <g className={state !== 'current' ? 'marker-pulse' : ''}>
            <circle cx="247" cy="173" r="12" fill="#d97706" fillOpacity=".13" />
            <circle cx="247" cy="173" r="5" fill="#d97706" stroke="#fff9ef" strokeWidth="2" />
            <path d="M247 152 V140" stroke="#b56a09" strokeDasharray="2 3" />
          </g>
          <g className={state === 'forecast' ? 'marker-pulse' : ''}>
            <circle cx="543" cy="278" r="12" fill="#d97706" fillOpacity=".13" />
            <circle cx="543" cy="278" r="5" fill="#d97706" stroke="#fff9ef" strokeWidth="2" />
            <path d="M543 257 V245" stroke="#b56a09" strokeDasharray="2 3" />
          </g>
          {state !== 'current' && (
            <g opacity=".85">
              <path d="M425 88 L455 83" stroke={copy.tint} strokeWidth="2" strokeDasharray="5 4" />
              <circle cx="455" cy="83" r="4" fill={copy.tint} />
            </g>
          )}
        </g>
        <g fill="#59645b" fontFamily="Inter, sans-serif" fontSize="10">
          <text x="74" y="408">12.4 ha</text>
          <text x="651" y="408">NORTH ↑</text>
          <text x="32" y="112" transform="rotate(-88 32 112)">FIELD BOUNDARY</text>
        </g>
      </svg>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-[#d6d9ce] bg-white/70 px-4 py-3 sm:px-6">
        <span className="flex items-center gap-2 text-[11px] text-[#59645b]"><span className="h-2 w-2 rounded-full bg-[#537953]" /> simulated crop</span>
        <span className="flex items-center gap-2 text-[11px] text-[#59645b]"><span className="h-2 w-2 rounded-full bg-[#d97706]" /> observation</span>
        <span className="ml-auto mono text-[10px] text-[#69736c]">illustrative spatial data</span>
      </div>
    </div>
  );
}