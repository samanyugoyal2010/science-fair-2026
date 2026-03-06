'use client'

import { useEffect, useRef, useState } from 'react'

function useInView(threshold = 0.15) {
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

function TransformerDiagram({ visible }: { visible: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2 py-4">
      {/* Input tokens */}
      <div className={`flex gap-2 transition-all duration-500 ${visible ? 'opacity-100' : 'opacity-0'}`}>
        {['T1', 'T2', 'T3', 'T4', 'T5'].map((t, i) => (
          <div
            key={t}
            className="w-9 h-9 rounded-md border border-border bg-secondary flex items-center justify-center text-xs font-mono text-muted-foreground"
            style={{ transitionDelay: `${i * 60}ms` }}
          >
            {t}
          </div>
        ))}
      </div>

      {/* Arrow down */}
      <div className="flex flex-col items-center gap-0">
        <div className="w-px h-4 bg-border" />
        <div className="text-border text-xs">▼</div>
      </div>

      {/* Single path box */}
      <div
        className={`w-52 h-14 rounded-lg border border-[oklch(0.65_0.18_30)] bg-[oklch(0.65_0.18_30/0.08)] flex items-center justify-center transition-all duration-700 ${
          visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
        }`}
        style={{ transitionDelay: '200ms' }}
      >
        <span className="text-[oklch(0.65_0.18_30)] text-xs font-semibold">
          Multi-Head Attention
        </span>
      </div>

      <div className="flex flex-col items-center gap-0">
        <div className="w-px h-4 bg-border" />
        <div className="text-border text-xs">▼</div>
      </div>

      <div
        className={`w-52 h-10 rounded-lg border border-[oklch(0.65_0.18_30/0.5)] bg-[oklch(0.65_0.18_30/0.05)] flex items-center justify-center transition-all duration-700 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        style={{ transitionDelay: '300ms' }}
      >
        <span className="text-[oklch(0.65_0.18_30)] text-xs">Feed-Forward Network</span>
      </div>

      <div className="flex flex-col items-center gap-0">
        <div className="w-px h-4 bg-border" />
        <div className="text-border text-xs">▼</div>
      </div>

      {/* Output */}
      <div
        className={`w-36 h-8 rounded-full border border-[oklch(0.65_0.18_30/0.4)] bg-[oklch(0.65_0.18_30/0.08)] flex items-center justify-center transition-all duration-700 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        style={{ transitionDelay: '400ms' }}
      >
        <span className="text-[oklch(0.65_0.18_30)] text-xs font-mono">Output</span>
      </div>

      <p className="text-muted-foreground text-xs mt-2 text-center max-w-[220px]">
        Single pathway — all processing flows through one sequential stack
      </p>
    </div>
  )
}

function HybridDiagram({ visible }: { visible: boolean }) {
  return (
    <div className="flex flex-col items-center gap-2 py-4">
      {/* Input tokens */}
      <div className={`flex gap-2 transition-all duration-500 ${visible ? 'opacity-100' : 'opacity-0'}`}>
        {['T1', 'T2', 'T3', 'T4', 'T5'].map((t, i) => (
          <div
            key={t}
            className="w-9 h-9 rounded-md border border-primary/40 bg-primary/10 flex items-center justify-center text-xs font-mono text-primary"
            style={{ transitionDelay: `${i * 60}ms` }}
          >
            {t}
          </div>
        ))}
      </div>

      {/* Split arrow */}
      <div className="flex items-end gap-10 h-8 relative">
        <div className="absolute left-1/2 -translate-x-1/2 bottom-0 w-px h-4 bg-primary/40" />
        <div
          className="absolute bottom-0 left-1/2 -translate-x-1/2 w-28 h-px bg-primary/30"
          style={{ bottom: '0px' }}
        />
      </div>

      {/* Two streams */}
      <div
        className={`flex gap-4 transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
        }`}
        style={{ transitionDelay: '200ms' }}
      >
        {/* SSM Stream */}
        <div className="flex flex-col items-center gap-1">
          <div className="border border-primary/50 bg-primary/10 rounded-lg px-4 py-2 text-center">
            <p className="text-primary text-xs font-bold">SSM Stream</p>
            <p className="text-muted-foreground text-[10px] mt-0.5">State-Space Model</p>
          </div>
          <div className="w-px h-3 bg-primary/30" />
          <div className="border border-primary/30 bg-primary/5 rounded px-3 py-1.5">
            <p className="text-primary/80 text-[10px] font-mono">ODE Memory Gate</p>
          </div>
        </div>

        {/* KAN Stream */}
        <div className="flex flex-col items-center gap-1">
          <div className="border border-primary/50 bg-primary/10 rounded-lg px-4 py-2 text-center">
            <p className="text-primary text-xs font-bold">KAN Stream</p>
            <p className="text-muted-foreground text-[10px] mt-0.5">Kolmogorov-Arnold</p>
          </div>
          <div className="w-px h-3 bg-primary/30" />
          <div className="border border-primary/30 bg-primary/5 rounded px-3 py-1.5">
            <p className="text-primary/80 text-[10px] font-mono">B-spline Activations</p>
          </div>
        </div>
      </div>

      {/* Converge arrows */}
      <div className="flex items-start gap-10 h-8 relative">
        <div className="absolute left-1/2 -translate-x-1/2 top-0 w-px h-4 bg-primary/40" />
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-28 h-px bg-primary/30" />
      </div>

      {/* Fusion box */}
      <div
        className={`w-52 h-12 rounded-lg border border-primary bg-primary/15 flex items-center justify-center transition-all duration-700 ${
          visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
        }`}
        style={{ transitionDelay: '400ms' }}
      >
        <span className="text-primary text-xs font-bold">Fusion Mechanism</span>
      </div>

      <div className="flex flex-col items-center">
        <div className="w-px h-4 bg-primary/40" />
        <div className="text-primary/50 text-xs">▼</div>
      </div>

      {/* Output */}
      <div
        className={`w-36 h-8 rounded-full border border-primary/60 bg-primary/15 flex items-center justify-center transition-all duration-700 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
        style={{ transitionDelay: '500ms' }}
      >
        <span className="text-primary text-xs font-mono font-bold">Output</span>
      </div>

      <p className="text-muted-foreground text-xs mt-2 text-center max-w-[220px]">
        Dual parallel streams fused — leverages complementary strengths
      </p>
    </div>
  )
}

export default function Architecture() {
  const { ref, visible } = useInView()

  return (
    <section id="architecture" className="py-24 px-4 sm:px-6 bg-card/30 border-y border-border" ref={ref}>
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <span className="text-xs font-mono text-primary uppercase tracking-widest">Design</span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
            Model Architecture
          </h2>
          <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
            Two architectures compared side-by-side — the traditional transformer single pathway
            versus the novel split-stream dual pathway.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Transformer card */}
          <div
            className={`border border-border rounded-xl bg-card p-6 transition-all duration-700 ${
              visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
            }`}
          >
            <div className="flex items-center gap-2 mb-6">
              <span className="w-2 h-2 rounded-full bg-[oklch(0.65_0.18_30)]" />
              <span className="text-sm font-semibold text-foreground">Transformer Baseline</span>
              <span className="ml-auto text-xs font-mono text-muted-foreground border border-border px-2 py-0.5 rounded">
                S33-T
              </span>
            </div>
            <TransformerDiagram visible={visible} />
          </div>

          {/* Hybrid card */}
          <div
            className={`border border-primary/30 rounded-xl bg-primary/5 p-6 transition-all duration-700 ${
              visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
            }`}
            style={{ transitionDelay: '150ms' }}
          >
            <div className="flex items-center gap-2 mb-6">
              <span className="w-2 h-2 rounded-full bg-primary animate-pulse-glow" />
              <span className="text-sm font-semibold text-primary">Hybrid S33</span>
              <span className="ml-auto text-xs font-mono text-primary border border-primary/30 px-2 py-0.5 rounded">
                S33-H
              </span>
            </div>
            <HybridDiagram visible={visible} />
          </div>
        </div>

        {/* Methodology strip */}
        <div
          className={`mt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '400ms' }}
        >
          {[
            { label: 'Matched Parameters', desc: 'Fair comparison ensured' },
            { label: 'Multiple Seeds', desc: 'Randomness controlled' },
            { label: 'Identical Training', desc: 'Same data & schedule' },
            { label: 'Multiple Ctx Lengths', desc: 'Generalization tested' },
          ].map((item) => (
            <div key={item.label} className="border border-border bg-card rounded-lg p-4">
              <p className="text-foreground text-xs font-semibold mb-1">{item.label}</p>
              <p className="text-muted-foreground text-xs">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
