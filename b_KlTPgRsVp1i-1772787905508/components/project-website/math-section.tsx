'use client'

import { useEffect, useRef, useState } from 'react'
import { InlineMath, BlockMath } from 'react-katex'

function useInView(threshold = 0.1) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const obs = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) setVisible(true)
    }, { threshold })
    if (ref.current) obs.observe(ref.current)
    return () => obs.disconnect()
  }, [threshold])
  return { ref, visible }
}

function GateAnimation({ visible }: { visible: boolean }) {
  const [step, setStep] = useState(0)
  useEffect(() => {
    if (!visible) return
    const interval = setInterval(() => setStep((s) => (s + 1) % 4), 1200)
    return () => clearInterval(interval)
  }, [visible])

  const steps = [
    { lambda: 0.1, label: 'Low \\lambda — preserve memory', mem: 90 },
    { lambda: 0.3, label: 'Medium \\lambda — partial update', mem: 65 },
    { lambda: 0.7, label: 'High \\lambda — important token!', mem: 35 },
    { lambda: 0.95, label: 'Max \\lambda — full rewrite', mem: 10 },
  ]

  const current = steps[step]

  return (
    <div className="space-y-3">
      <p className="text-muted-foreground text-xs font-mono mb-4">
        Selective gate <InlineMath math="\lambda(t)" /> controls memory update speed
      </p>

      {/* Lambda slider visual */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-muted-foreground w-12">
          <InlineMath math="\lambda =" />
        </span>
        <div className="flex-1 bg-secondary rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all duration-700"
            style={{ width: `${current.lambda * 100}%` }}
          />
        </div>
        <span className="text-xs font-mono text-primary w-10 text-right">{current.lambda}</span>
      </div>

      {/* Memory retention */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono text-muted-foreground w-12">Memory</span>
        <div className="flex-1 bg-secondary rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-primary/60 rounded-full transition-all duration-700"
            style={{ width: `${current.mem}%` }}
          />
        </div>
        <span className="text-xs font-mono text-muted-foreground w-10 text-right">{current.mem}%</span>
      </div>

      {/* Label */}
      <p className="text-xs text-primary font-mono border border-primary/20 bg-primary/5 px-3 py-2 rounded-lg transition-all duration-300">
        {current.label.replace('\\lambda', 'λ')}
      </p>

      {/* Step dots */}
      <div className="flex gap-1.5 pt-1">
        {steps.map((_, i) => (
          <div
            key={i}
            className={`w-1.5 h-1.5 rounded-full transition-colors duration-300 ${
              i === step ? 'bg-primary' : 'bg-border'
            }`}
          />
        ))}
      </div>
    </div>
  )
}

const chainSteps = [
  {
    label: 'ODE',
    eq: '\\frac{ds}{dt} = -\\lambda(t)\\,s(t) + \\lambda(t)\\,b(t)',
    note: 'Continuous-time memory evolution',
    highlight: false,
  },
  {
    label: 'Integral',
    eq: 's(t) = e^{-\\int \\lambda}\\,s(0) + \\int e^{-\\int \\lambda}\\lambda(\\tau)\\,b(\\tau)\\,d\\tau',
    note: 'Closed-form solution',
    highlight: false,
  },
  {
    label: 'Discretize',
    eq: 's_t = e^{-\\lambda_t} s_{t-1} + (1-e^{-\\lambda_t})\\,b_t',
    note: 'One step at a time',
    highlight: false,
  },
  {
    label: 'Reparameterize',
    eq: '\\alpha_t = \\sigma(\\delta_t) \\approx 1 - e^{-\\lambda_t}',
    note: 'Learnable gate via sigmoid',
    highlight: false,
  },
  {
    label: 'Code',
    eq: 'h = (1 - \\alpha) \\cdot h + \\alpha \\cdot b_{\\text{proj}}',
    note: '44 lines of Python',
    highlight: true,
  },
]

export default function MathSection() {
  const { ref, visible } = useInView()

  return (
    <section id="math" className="py-24 px-4 sm:px-6 max-w-6xl mx-auto" ref={ref}>
      <div className="mb-12">
        <span className="text-xs font-mono text-primary uppercase tracking-widest">Theory</span>
        <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
          The SSM ODE — Novel Contribution
        </h2>
        <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
          A first-order ODE governs how the model{"'"}s memory state evolves. Unlike fixed dynamics,
          the gate <InlineMath math="\lambda(t)" /> is computed from the input — making it input-adaptive.
        </p>
      </div>

      {/* Main equation hero */}
      <div
        className={`rounded-2xl border border-primary/30 bg-card p-8 mb-8 text-center transition-all duration-700 ${
          visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
        }`}
      >
        <p className="text-xs font-mono text-muted-foreground mb-6 uppercase tracking-widest">
          Core Equation
        </p>
        <div className="overflow-x-auto flex justify-center">
          <div className="text-2xl sm:text-3xl text-primary py-2 [&_.katex]:text-primary [&_.katex-html]:text-primary">
            <BlockMath math="\frac{ds}{dt} = -\lambda(t)\,s(t) + \lambda(t)\,b(t)" />
          </div>
        </div>
        <div className="mt-6 flex flex-wrap justify-center gap-6">
          {[
            { sym: 's(t)', latexSym: 's(t)', desc: 'Memory state — 64-dim notepad carried through sequence' },
            { sym: 'b(t)', latexSym: 'b(t)', desc: 'Current input at position t — new information' },
            { sym: 'λ(t)', latexSym: '\\lambda(t)', desc: 'Selective gate — computed from input, not fixed' },
          ].map((item) => (
            <div key={item.sym} className="flex items-start gap-2 max-w-[180px] text-left">
              <span className="text-primary text-sm font-bold flex-shrink-0 [&_.katex]:text-primary">
                <InlineMath math={item.latexSym} />
              </span>
              <span className="text-muted-foreground text-xs leading-relaxed">{item.desc}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Forgetting & Writing */}
        <div
          className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '150ms' }}
        >
          <h3 className="text-foreground font-bold text-base mb-4">The Two Terms</h3>
          <div className="space-y-4">
            <div className="flex gap-3">
              <div className="mt-1 w-1 flex-shrink-0 rounded-full bg-[#e07b45]" />
              <div>
                <p className="text-foreground text-sm font-semibold mb-1 [&_.katex]:text-foreground">
                  <InlineMath math="-\lambda(t) \cdot s(t)" />
                </p>
                <p className="text-muted-foreground text-xs leading-relaxed">
                  <span className="text-foreground font-medium">Forgetting term.</span> Pulls memory
                  toward zero. Higher <InlineMath math="\lambda" /> = faster exponential decay — same math as radioactive decay
                  or capacitor discharge.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="mt-1 w-1 flex-shrink-0 rounded-full bg-primary" />
              <div>
                <p className="text-foreground text-sm font-semibold mb-1 [&_.katex]:text-foreground">
                  <InlineMath math="+\lambda(t) \cdot b(t)" />
                </p>
                <p className="text-muted-foreground text-xs leading-relaxed">
                  <span className="text-foreground font-medium">Writing term.</span> Pushes memory
                  toward current input. Higher <InlineMath math="\lambda" /> = more aggressively overwrites with new
                  information.
                </p>
              </div>
            </div>
            <div className="border-t border-border pt-4 mt-2">
              <p className="text-muted-foreground text-xs leading-relaxed">
                Both terms share the same <InlineMath math="\lambda" />, creating a single knob per timestep:
                high <InlineMath math="\lambda" /> = forget fast, write fast. Low <InlineMath math="\lambda" /> = forget slow, preserve memory.
              </p>
            </div>
          </div>
        </div>

        {/* Gate animation */}
        <div
          className={`border border-primary/20 bg-primary/5 rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '250ms' }}
        >
          <h3 className="text-primary font-bold text-base mb-4">Live Gate Simulation</h3>
          <GateAnimation visible={visible} />
        </div>
      </div>

      {/* Math → Code chain */}
      <div
        className={`mt-8 border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
        style={{ transitionDelay: '350ms' }}
      >
        <h3 className="text-foreground font-bold text-base mb-6">
          Chain: Math <span className="text-muted-foreground font-normal">→</span> Code
        </h3>
        <div className="flex flex-col sm:flex-row flex-wrap gap-2 items-start">
          {chainSteps.map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              <div
                className={`rounded-lg border p-3 min-w-[140px] ${
                  step.highlight
                    ? 'border-primary/50 bg-primary/10'
                    : 'border-border bg-secondary/50'
                }`}
              >
                <p
                  className={`text-[10px] font-mono uppercase tracking-wider mb-2 ${
                    step.highlight ? 'text-primary' : 'text-muted-foreground'
                  }`}
                >
                  {step.label}
                </p>
                <div
                  className={`text-xs leading-tight [&_.katex]:text-[0.7rem] ${
                    step.highlight ? '[&_.katex]:!text-primary' : '[&_.katex]:!text-foreground'
                  }`}
                >
                  <InlineMath math={step.eq} />
                </div>
                <p className="text-muted-foreground text-[10px] mt-2 leading-snug">
                  {step.note}
                </p>
              </div>
              {i < chainSteps.length - 1 && (
                <span className="text-muted-foreground text-lg hidden sm:block">→</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
