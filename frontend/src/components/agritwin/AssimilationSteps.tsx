import { Check, Database, Eye, FlaskConical, GitMerge, Sprout } from 'lucide-react';
import { assimilationSteps } from '@/data/demo';

type AssimilationStepsProps = {
  activeStep: number;
  onChange: (step: number) => void;
};

const icons = [Sprout, Eye, GitMerge, FlaskConical, Database, Check];

export function AssimilationSteps({ activeStep, onChange }: AssimilationStepsProps) {
  return (
    <div className="relative">
      <div className="absolute left-[8%] right-[8%] top-[22px] hidden h-px bg-[#cbd7c7] lg:block" />
      <div className="grid grid-cols-2 gap-y-8 sm:grid-cols-3 lg:grid-cols-6 lg:gap-3">
        {assimilationSteps.map((step, index) => {
          const Icon = icons[index];
          const active = activeStep === index;
          return (
            <button
              type="button"
              data-testid={`button-assimilation-step-${index + 1}`}
              key={step.id}
              onClick={() => onChange(index)}
              className={`group relative text-left lg:text-center ${active ? 'text-[#1b3022]' : 'text-[#7b837c]'}`}
            >
              <span className={`relative z-10 mx-0 flex h-11 w-11 items-center justify-center border transition-all duration-300 lg:mx-auto ${active ? 'border-[#1b3022] bg-[#1b3022] text-white' : 'border-[#bfcabd] bg-[#faf9f6] group-hover:border-[#6b876d] group-hover:text-[#1b3022]'}`}>
                <Icon size={17} strokeWidth={1.5} />
              </span>
              <span className="mt-3 block">
                <span className="block text-sm font-semibold">{step.id} {step.title}</span>
                <span className="mt-1 block text-[11px] text-[#7b837c]">{step.detail}</span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}