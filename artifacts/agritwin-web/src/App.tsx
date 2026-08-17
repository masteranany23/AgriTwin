import { useState } from 'react';
import type { ReactNode } from 'react';
import { ArrowDownRight, ArrowUpRight, BarChart3, ChevronRight, Leaf, Menu, MoveRight, Satellite, X } from 'lucide-react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TooltipProvider } from '@/components/ui/tooltip';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { FieldTwin } from '@/components/agritwin/FieldTwin';
import { AssimilationSteps } from '@/components/agritwin/AssimilationSteps';
import { TrajectoryChart } from '@/components/agritwin/TrajectoryChart';
import { demoField, type TwinState } from '@/data/demo';
import { Route, Switch, Router as WouterRouter, useLocation } from 'wouter';
import NotFound from '@/pages/not-found';

const queryClient = new QueryClient();

function MetricCard({ label, value, note, accent, trend }: { label: string; value: string; note: string; accent: string; trend?: string }) {
  return (
    <article className="group border border-[#e1e4de] bg-white p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-[#b7c9b7] hover:shadow-[0_14px_34px_rgba(27,48,34,.07)] sm:p-6" data-testid={`card-metric-${label.toLowerCase().replaceAll(' ', '-')}`}>
      <div className="flex items-start justify-between">
        <p className="label-caps text-[#69736c]">{label}</p>
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: accent }} />
      </div>
      <div className="mt-7 flex items-end justify-between gap-3">
        <p className="mono text-[26px] font-medium tracking-[-.04em] text-[#1b3022]" data-testid={`text-metric-${label.toLowerCase().replaceAll(' ', '-')}`}>{value}</p>
        {trend && <span className="flex items-center gap-1 text-[10px] font-semibold text-[#668167]"><ArrowUpRight size={13} /> {trend}</span>}
      </div>
      <p className="mt-2 text-xs text-[#788078]">{note}</p>
    </article>
  );
}

function Nav() {
  const [open, setOpen] = useState(false);
  const links = [
    { label: 'The Twin', href: '#twin' },
    { label: 'How It Works', href: '#science' },
    { label: 'Trajectory', href: '#trajectory' },
  ];
  return (
    <header className="sticky top-0 z-30 border-b border-[#e3e5df]/90 bg-[#faf9f6]/90 backdrop-blur-md">
      <div className="mx-auto flex h-[72px] max-w-[1280px] items-center justify-between px-5 sm:px-8 lg:px-12">
        <a href="#top" className="flex items-center gap-3" data-testid="link-brand">
          <span className="flex h-8 w-8 items-center justify-center border border-[#1b3022] bg-[#1b3022] text-[#faf9f6]">
            <Leaf size={16} strokeWidth={1.7} />
          </span>
          <span className="font-semibold tracking-[-.04em] text-[#1b3022]">AgriTwin</span>
          <span className="mono ml-1 border-l border-[#d8ded5] pl-3 text-[9px] tracking-[.12em] text-[#7d857f]">v0.1</span>
        </a>
        <nav className="hidden items-center gap-8 md:flex" aria-label="Primary navigation">
          {links.map((link) => <a key={link.href} href={link.href} className="text-[12px] text-[#68716a] transition-colors hover:text-[#1b3022]" data-testid={`link-nav-${link.label.toLowerCase().replaceAll(' ', '-')}`}>{link.label}</a>)}
        </nav>
        <div className="hidden items-center gap-4 md:flex">
          <span className="flex items-center gap-2 text-[10px] text-[#69736c]"><span className="h-1.5 w-1.5 rounded-full bg-[#6f9771]" /> DEMONSTRATION MODE</span>
          <a href="#twin" className="flex items-center gap-2 bg-[#1b3022] px-4 py-2.5 text-[11px] font-semibold text-white transition-colors hover:bg-[#2d4935]" data-testid="link-open-twin">Open field twin <MoveRight size={14} /></a>
        </div>
        <button type="button" className="p-2 text-[#1b3022] md:hidden" onClick={() => setOpen((value) => !value)} aria-label="Toggle navigation" data-testid="button-toggle-navigation">
          {open ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>
      {open && (
        <nav className="border-t border-[#e3e5df] bg-[#faf9f6] px-5 py-4 md:hidden" aria-label="Mobile navigation">
          {links.map((link) => <a key={link.href} href={link.href} onClick={() => setOpen(false)} className="block border-b border-[#e3e5df] py-3 text-sm text-[#465149]" data-testid={`link-mobile-${link.label.toLowerCase().replaceAll(' ', '-')}`}>{link.label}</a>)}
        </nav>
      )}
    </header>
  );
}

function Hero() {
  return (
    <section id="top" className="mx-auto max-w-[1280px] px-5 pb-20 pt-20 sm:px-8 sm:pb-28 sm:pt-28 lg:px-12 lg:pt-32">
      <div className="grid items-end gap-12 lg:grid-cols-[1.14fr_.86fr] lg:gap-20">
        <div className="reveal">
          <p className="label-caps flex items-center gap-3 text-[#6b866b]"><span className="h-px w-8 bg-[#6b866b]" /> AGRICULTURAL DIGITAL TWIN</p>
          <h1 className="mt-7 max-w-[680px] text-[clamp(2.7rem,6.2vw,5.35rem)] font-semibold leading-[.99] tracking-[-.075em] text-[#1b3022]">A Digital Twin<br /><span className="text-[#71856d]">for Every Field.</span></h1>
          <p className="mt-8 max-w-[570px] text-[16px] leading-7 text-[#5e675f] sm:text-[18px]">AgriTwin combines physics-based crop simulation, real-world observations and sequential data assimilation to understand field conditions and forecast what happens next.</p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <a href="#twin" className="flex items-center gap-3 bg-[#1b3022] px-5 py-3.5 text-[12px] font-semibold text-white transition-colors hover:bg-[#2d4935]" data-testid="link-hero-explore">Explore the Digital Twin <ArrowDownRight size={15} /></a>
            <a href="#science" className="flex items-center gap-2 border border-[#1b3022] px-5 py-3.5 text-[12px] font-semibold text-[#1b3022] transition-colors hover:bg-[#eef2eb]" data-testid="link-hero-how-it-works">How it works <ChevronRight size={15} /></a>
          </div>
        </div>
        <div className="reveal reveal-delay-2 relative min-h-[190px] border-l border-[#d9ded7] pl-7 sm:pl-10">
          <span className="absolute -left-[4px] top-0 h-2 w-2 rounded-full bg-[#d97706]" />
          <p className="label-caps text-[#8b928c]">A research instrument for</p>
          <div className="mt-8 grid max-w-[390px] grid-cols-2 gap-x-8 gap-y-6">
            {['Crop researchers', 'Climate programs', 'Public institutions', 'Field operations'].map((item, index) => (
              <div key={item} className="border-t border-[#dfe3dc] pt-3">
                <p className="mono text-[10px] text-[#a0a69f]">0{index + 1}</p>
                <p className="mt-2 text-[13px] font-medium text-[#4a554d]">{item}</p>
              </div>
            ))}
          </div>
          <p className="absolute bottom-0 left-7 max-w-[330px] text-[11px] leading-5 text-[#7a827b] sm:left-10">Not a dashboard. A clear view of the state beneath the canopy.</p>
        </div>
      </div>
    </section>
  );
}

function TwinSection({ state, onStateChange }: { state: TwinState; onStateChange: (state: TwinState) => void }) {
  const stateButtons: { id: TwinState; label: string }[] = [
    { id: 'current', label: 'Current state' },
    { id: 'assimilated', label: 'Assimilated' },
    { id: 'forecast', label: 'Forecast' },
  ];
  return (
    <section id="twin" className="bg-[#f0f3ed] py-20 sm:py-28">
      <div className="mx-auto max-w-[1280px] px-5 sm:px-8 lg:px-12">
        <div className="mb-8 flex flex-col justify-between gap-5 sm:flex-row sm:items-end">
          <div>
            <p className="label-caps text-[#6b866b]">01 / FIELD DIGITAL TWIN</p>
            <h2 className="mt-3 text-2xl font-semibold tracking-[-.05em] text-[#1b3022] sm:text-3xl">Read the field as a living system.</h2>
          </div>
          <div className="flex border border-[#cbd7c7] bg-[#faf9f6] p-1" role="tablist" aria-label="Twin visualization state">
            {stateButtons.map((button) => <button key={button.id} type="button" role="tab" aria-selected={state === button.id} onClick={() => onStateChange(button.id)} className={`px-3 py-2 text-[10px] font-semibold transition-colors sm:px-4 ${state === button.id ? 'bg-[#1b3022] text-white' : 'text-[#69736c] hover:text-[#1b3022]'}`} data-testid={`button-state-${button.id}`}>{button.label}</button>)}
          </div>
        </div>
        <div className="grid gap-4 lg:grid-cols-[1.55fr_.45fr]">
          <FieldTwin state={state} />
          <aside className="flex flex-col justify-between border border-[#dfe3dc] bg-white p-5 sm:p-6">
            <div>
              <div className="flex items-center justify-between">
                <p className="label-caps text-[#69736c]">Field profile</p>
                <span className="mono text-[10px] text-[#8b928c]">LOCAL DEMO</span>
              </div>
              <div className="mt-7 border-y border-[#e1e4de] py-5">
                <p className="text-lg font-semibold tracking-[-.04em] text-[#1b3022]" data-testid="text-field-name">{demoField.name}</p>
                <div className="mt-4 flex items-center justify-between text-xs text-[#69736c]"><span>Crop</span><span className="font-semibold text-[#465149]">{demoField.crop}</span></div>
                <div className="mt-2 flex items-center justify-between text-xs text-[#69736c]"><span>Simulation day</span><span className="mono text-[#465149]">{demoField.day}</span></div>
                <div className="mt-2 flex items-center justify-between text-xs text-[#69736c]"><span>Area</span><span className="mono text-[#465149]">12.4 ha</span></div>
              </div>
              <div className="mt-6 flex gap-3">
                <div className="flex h-8 w-8 items-center justify-center border border-[#e5e7df] text-[#6e876d]"><Satellite size={15} /></div>
                <div><p className="text-xs font-semibold text-[#465149]">Observation layer</p><p className="mt-1 text-[11px] leading-4 text-[#7a827b]">2 spatial samples · Sentinel-2</p></div>
              </div>
              <div className="mt-6 space-y-2 border-t border-[#e1e4de] pt-5">
                <div className="flex items-center justify-between text-[11px]"><span className="text-[#69736c]">Observed LAI</span><span className="mono font-medium text-[#d97706]" data-testid="text-observed-lai">{demoField.observation.lai} <span className="text-[9px] text-[#8b928c]">({demoField.observation.confidence})</span></span></div>
                <div className="flex items-center justify-between text-[11px]"><span className="text-[#69736c]">Assimilated LAI</span><span className="mono font-medium text-[#668167]" data-testid="text-assimilated-lai">{demoField.assimilated.lai}</span></div>
                <div className="flex items-center justify-between text-[11px]"><span className="text-[#69736c]">Forecast LAI</span><span className="mono font-medium text-[#3b82f6]" data-testid="text-forecast-lai">{demoField.forecast.lai}</span></div>
                <div className="flex items-center justify-between text-[11px]"><span className="text-[#69736c]">Projected yield</span><span className="mono font-medium text-[#465149]" data-testid="text-forecast-yield">{demoField.forecast.yield}</span></div>
              </div>
            </div>
            <p className="mt-8 border-t border-[#e1e4de] pt-4 text-[10px] leading-4 text-[#8b928c]">All values shown are illustrative demonstration values, not live measurements.</p>
          </aside>
        </div>
      </div>
    </section>
  );
}

function StatusSection() {
  return (
    <section className="mx-auto max-w-[1280px] px-5 py-20 sm:px-8 sm:py-28 lg:px-12">
      <div className="flex flex-col justify-between gap-4 border-b border-[#dfe3dc] pb-6 sm:flex-row sm:items-end">
        <div><p className="label-caps text-[#6b866b]">02 / FIELD STATUS</p><h2 className="mt-3 text-2xl font-semibold tracking-[-.05em] text-[#1b3022] sm:text-3xl">A concise state estimate.</h2></div>
        <p className="max-w-[300px] text-xs leading-5 text-[#788078]">Current state, expressed in the variables that matter for crop development.</p>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label="Leaf Area Index" value={demoField.current.lai} note="Current canopy surface" accent="#68876c" trend="+0.18" />
        <MetricCard label="Soil Moisture" value={demoField.current.moisture} note="Volumetric water content" accent="#3b82f6" />
        <MetricCard label="Total Biomass" value={demoField.current.biomass} note="Above-ground estimate" accent="#d97706" trend="+0.32" />
        <MetricCard label="Development Stage" value={`DVS ${demoField.current.dvs}`} note="Phenological progress" accent="#8fa382" />
      </div>
    </section>
  );
}

function ScienceSection({ activeStep, setActiveStep }: { activeStep: number; setActiveStep: (step: number) => void }) {
  return (
    <section id="science" className="border-y border-[#e0e4dc] bg-[#f0f3ed] py-20 sm:py-28">
      <div className="mx-auto max-w-[1280px] px-5 sm:px-8 lg:px-12">
        <div className="grid gap-12 lg:grid-cols-[.7fr_1.3fr] lg:gap-20">
          <div>
            <p className="label-caps text-[#6b866b]">03 / STATE ASSIMILATION</p>
            <h2 className="mt-5 max-w-[390px] text-3xl font-semibold leading-tight tracking-[-.06em] text-[#1b3022] sm:text-4xl">From observation to a better estimate.</h2>
            <p className="mt-5 max-w-[370px] text-sm leading-6 text-[#657067]">AgriTwin continuously reconciles what the model expects with what the field reveals. Each observation narrows uncertainty without losing the physics.</p>
            <div className="mt-9 flex items-center gap-3 border-l-2 border-[#d97706] pl-4"><span className="mono text-[10px] text-[#8b928c]">ENKF</span><span className="text-xs text-[#4f5c52]">Sequential data assimilation</span></div>
          </div>
          <AssimilationSteps activeStep={activeStep} onChange={setActiveStep} />
        </div>
      </div>
    </section>
  );
}

function TrajectorySection() {
  return (
    <section id="trajectory" className="mx-auto max-w-[1280px] px-5 py-20 sm:px-8 sm:py-28 lg:px-12">
      <div className="grid gap-10 lg:grid-cols-[.72fr_1.28fr] lg:gap-20">
        <div>
          <p className="label-caps text-[#6b866b]">04 / ASSIMILATION TRAJECTORY</p>
          <h2 className="mt-5 max-w-[360px] text-3xl font-semibold leading-tight tracking-[-.06em] text-[#1b3022] sm:text-4xl">The forecast remembers what the field said.</h2>
          <p className="mt-5 max-w-[360px] text-sm leading-6 text-[#657067]">The open-loop model drifts from observed conditions. Assimilation brings the twin back into alignment, then carries that improved state forward.</p>
          <div className="mt-8 space-y-3">
            <div className="flex items-center gap-3 text-xs text-[#536056]"><span className="h-0.5 w-7 bg-[#1b3022]" /> Open-loop simulation</div>
            <div className="flex items-center gap-3 text-xs text-[#536056]"><span className="w-7 border-t-2 border-dashed border-[#8fa382]" /> Assimilated simulation</div>
            <div className="flex items-center gap-3 text-xs text-[#536056]"><span className="h-2 w-2 rounded-full bg-[#d97706] ring-2 ring-[#f8ead4]" /> Observations</div>
          </div>
        </div>
        <div className="border border-[#e0e4dc] bg-white p-4 sm:p-7">
          <div className="mb-5 flex items-start justify-between border-b border-[#e5e7df] pb-4">
            <div><p className="label-caps text-[#69736c]">LAI / SIMULATION DAY</p><p className="mt-2 mono text-xl text-[#1b3022]">0.64 — 3.51</p></div>
            <span className="flex items-center gap-1 text-[10px] text-[#6b866b]"><BarChart3 size={14} /> demo trajectory</span>
          </div>
          <TrajectoryChart />
          <div className="mt-3 flex justify-between text-[10px] text-[#8b928c]"><span>Illustrative values</span><span>Forecast window → day 63</span></div>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="bg-[#1b3022] text-[#e8efe6]">
      <div className="mx-auto max-w-[1280px] px-5 py-12 sm:px-8 lg:px-12">
        <div className="flex flex-col justify-between gap-10 sm:flex-row">
          <div><div className="flex items-center gap-3"><span className="flex h-8 w-8 items-center justify-center border border-[#819986]"><Leaf size={16} /></span><span className="font-semibold tracking-[-.04em]">AgriTwin</span></div><p className="mt-5 max-w-[290px] text-xs leading-5 text-[#aab9aa]">A quiet, credible digital twin for agricultural research and climate intelligence.</p></div>
          <div className="grid grid-cols-2 gap-x-12 gap-y-3 text-xs text-[#b9c6ba]"><a href="#twin" className="hover:text-white" data-testid="link-footer-twin">The Twin</a><a href="#science" className="hover:text-white" data-testid="link-footer-science">How It Works</a><a href="#trajectory" className="hover:text-white" data-testid="link-footer-trajectory">Trajectory</a><a href="#top" className="hover:text-white" data-testid="link-footer-top">Back to top</a></div>
        </div>
        <div className="mt-12 flex flex-col justify-between gap-3 border-t border-[#36503c] pt-5 text-[10px] text-[#94a695] sm:flex-row"><span>AGRITWIN / SCIENTIFIC SYSTEMS / V0.1</span><span>Illustrative demonstration interface · no live field measurements</span></div>
      </div>
    </footer>
  );
}

function Home() {
  const [state, setState] = useState<TwinState>('current');
  const [activeStep, setActiveStep] = useState(0);
  return (
    <div className="grain min-h-[100dvh] overflow-x-hidden bg-[#faf9f6]">
      <Nav />
      <main>
        <Hero />
        <TwinSection state={state} onStateChange={setState} />
        <StatusSection />
        <ScienceSection activeStep={activeStep} setActiveStep={setActiveStep} />
        <TrajectorySection />
      </main>
      <Footer />
    </div>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;