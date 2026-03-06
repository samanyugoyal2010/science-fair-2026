'use client'

import { useEffect, useRef, useState } from 'react'

function useInView(threshold = 0.2) {
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

const TOTAL_TOKENS = 128

function TokenStream({
  model,
  color,
  speed,
  visible,
  label,
  badge,
}: {
  model: string
  color: string
  speed: number
  visible: boolean
  label: string
  badge?: string
}) {
  const [processed, setProcessed] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!visible) return
    // Reset first
    setProcessed(0)
    setIsRunning(false)
    if (intervalRef.current) clearInterval(intervalRef.current)

    const delay = model === 'hybrid' ? 200 : 600
    const timeout = setTimeout(() => {
      setIsRunning(true)
      intervalRef.current = setInterval(() => {
        setProcessed((p) => {
          if (p >= TOTAL_TOKENS) {
            clearInterval(intervalRef.current!)
            setIsRunning(false)
            return p
          }
          return p + 1
        })
      }, speed)
    }, delay)

    return () => {
      clearTimeout(timeout)
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [visible, model, speed])

  const pct = Math.round((processed / TOTAL_TOKENS) * 100)

  // We show a compact grid of 64 cells, each cell represents 2 tokens
  const DISPLAY_CELLS = 64
  const processedCells = Math.round((processed / TOTAL_TOKENS) * DISPLAY_CELLS)

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
          <span className="text-foreground font-semibold text-sm">{label}</span>
          {badge && (
            <span
              className="text-xs font-mono px-2 py-0.5 rounded-full border"
              style={{ color, borderColor: color + '55', backgroundColor: color + '15' }}
            >
              {badge}
            </span>
          )}
        </div>
        <span className="font-mono text-sm font-bold" style={{ color }}>
          {processed}<span className="text-muted-foreground font-normal text-xs"> / {TOTAL_TOKENS}</span>
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-secondary rounded-full h-2 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-100"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>

      {/* Compact token grid — 64 cells representing 128 tokens (2 per cell) */}
      <div
        className="grid gap-0.5"
        style={{ gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}
        aria-label={`Token processing grid — ${processed} of ${TOTAL_TOKENS} tokens processed`}
      >
        {Array.from({ length: DISPLAY_CELLS }, (_, i) => (
          <div
            key={i}
            className="aspect-square rounded-sm transition-all duration-75 border"
            style={{
              backgroundColor: i < processedCells ? color + '25' : 'transparent',
              borderColor: i < processedCells ? color + '70' : 'rgba(40,44,68,0.8)',
            }}
          />
        ))}
      </div>

      {/* Status */}
      <div
        className="text-xs font-mono px-3 py-2 rounded border"
        style={{
          borderColor: color + '30',
          backgroundColor: color + '08',
          color,
        }}
      >
        {processed >= TOTAL_TOKENS
          ? `Done — all ${TOTAL_TOKENS} tokens processed`
          : isRunning
          ? `Processing token ${processed + 1} of ${TOTAL_TOKENS}...`
          : 'Waiting to start...'}
      </div>
    </div>
  )
}

export default function SpeedComparison() {
  const { ref, visible } = useInView()
  const [replayKey, setReplayKey] = useState(0)
  const [replaying, setReplaying] = useState(false)

  const handleReplay = () => {
    setReplaying(true)
    setReplayKey((k) => k + 1)
    setTimeout(() => setReplaying(false), 100)
  }

  return (
    <section
      id="comparison"
      className="py-24 px-4 sm:px-6 bg-card/20 border-y border-border"
      ref={ref}
    >
      <div className="max-w-6xl mx-auto">
        <div className="mb-12">
          <span className="text-xs font-mono text-primary uppercase tracking-widest">
            Visual Demo
          </span>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
            Split-Stream Hybrid vs Transformer Baseline
          </h2>
          <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
            Illustrative animation at a 128-token context window. The hybrid model{"'"}s selective
            memory gate enables more efficient processing, achieving lower perplexity faster than
            the transformer baseline.
          </p>
        </div>

        <div
          className={`grid grid-cols-1 md:grid-cols-2 gap-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
        >
          {/* Hybrid model */}
          <div className="border border-primary/30 bg-primary/5 rounded-xl p-6">
            <TokenStream
              key={`hybrid-${replayKey}`}
              model="hybrid"
              color="#3dd9c0"
              speed={30}
              visible={visible && !replaying}
              label="Split-Stream Hybrid"
              badge="Faster"
            />
          </div>

          {/* Transformer baseline */}
          <div className="border border-border bg-card rounded-xl p-6">
            <TokenStream
              key={`transformer-${replayKey}`}
              model="transformer"
              color="#e07b45"
              speed={70}
              visible={visible && !replaying}
              label="Transformer Baseline"
            />
          </div>
        </div>

        {/* Replay button */}
        <div className="flex justify-center mt-6">
          <button
            onClick={handleReplay}
            className="border border-border text-muted-foreground hover:text-primary hover:border-primary/50 text-sm font-mono px-4 py-2 rounded-lg transition-colors"
          >
            Replay Animation
          </button>
        </div>

        {/* Advantage callout */}
        <div
          className={`mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '200ms' }}
        >
          {[
            {
              title: 'Selective Gate',
              desc: 'λ(t) adapts per token — important tokens update memory faster, predictable tokens coast',
            },
            {
              title: 'Dual Stream Fusion',
              desc: 'SSM long-range memory + KAN non-linear functions combine for richer representations',
            },
            {
              title: 'Lower Perplexity',
              desc: 'Consistently achieves lower val_ppl than the transformer across all context lengths',
            },
          ].map((item) => (
            <div key={item.title} className="border border-primary/20 bg-primary/5 rounded-xl p-5">
              <h4 className="font-bold text-sm mb-2 text-primary">{item.title}</h4>
              <p className="text-muted-foreground text-xs leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>

        {/* Context length note */}
        <div
          className={`mt-4 flex items-start gap-3 border border-border bg-card rounded-xl p-4 transition-all duration-700 ${
            visible ? 'opacity-100' : 'opacity-0'
          }`}
          style={{ transitionDelay: '350ms' }}
        >
          <span className="text-primary font-mono text-xs mt-0.5 flex-shrink-0">128 ctx</span>
          <p className="text-muted-foreground text-xs leading-relaxed">
            This demonstration illustrates the relative processing advantage of the Split-Stream Hybrid
            at a 128-token context window. The hybrid{"'"}s ODE-derived gating mechanism adapts to
            sequence structure while the transformer processes all tokens with full quadratic
            attention, causing higher memory overhead as context grows.
          </p>
        </div>
      </div>
    </section>
  )
}
