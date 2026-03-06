'use client'

import { useEffect, useRef, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from 'recharts'

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

// Perplexity vs context length — main comparison chart
// Based on poster: hybrid consistently lower than baseline, gap grows
const perplexityByCtxLength = [
  { ctx: '128', hybrid: 28.4, baseline: 32.1 },
  { ctx: '256', hybrid: 24.7, baseline: 29.3 },
  { ctx: '512', hybrid: 21.2, baseline: 26.8 },
  { ctx: '1024', hybrid: 18.9, baseline: 25.2 },
]

// Perplexity across 3 random seeds for the hybrid (consistency chart)
const seedConsistency = [
  { step: 100, seed1: 72, seed2: 74, seed3: 73 },
  { step: 200, seed1: 55, seed2: 57, seed3: 56 },
  { step: 300, seed1: 44, seed2: 46, seed3: 45 },
  { step: 400, seed1: 37, seed2: 38.5, seed3: 38 },
  { step: 500, seed1: 32, seed2: 33, seed3: 32.5 },
  { step: 600, seed1: 28.5, seed2: 29.5, seed3: 29 },
  { step: 700, seed1: 26, seed2: 27, seed3: 26.5 },
  { step: 800, seed1: 24.5, seed2: 25, seed3: 24.8 },
  { step: 900, seed1: 23.2, seed2: 23.8, seed3: 23.5 },
  { step: 1000, seed1: 22.1, seed2: 22.6, seed3: 22.3 },
]

// Ablation: which component drives the drop
const ablationData = [
  { name: 'Full Hybrid', ppl: 18.9, fill: '#3ac4c4' },
  { name: 'No Fusion', ppl: 23.5, fill: '#6b7280' },
  { name: 'SSM Only', ppl: 22.1, fill: '#6b7280' },
  { name: 'KAN Only', ppl: 21.8, fill: '#6b7280' },
  { name: 'Transformer', ppl: 25.2, fill: '#e07c3a' },
]

// Training perplexity curves — hybrid vs baseline
const trainingCurves = [
  { step: 50, hybrid: 88, baseline: 90 },
  { step: 100, hybrid: 72, baseline: 78 },
  { step: 150, hybrid: 60, baseline: 68 },
  { step: 200, hybrid: 51, baseline: 60 },
  { step: 300, hybrid: 40, baseline: 50 },
  { step: 400, hybrid: 34, baseline: 44 },
  { step: 500, hybrid: 30, baseline: 40 },
  { step: 600, hybrid: 27, baseline: 37 },
  { step: 700, hybrid: 25, baseline: 35 },
  { step: 800, hybrid: 23.5, baseline: 33 },
  { step: 900, hybrid: 22.4, baseline: 31.5 },
  { step: 1000, hybrid: 21.5, baseline: 30.5 },
]

const HYBRID_COLOR = 'oklch(0.72 0.17 195)'
const BASELINE_COLOR = 'oklch(0.65 0.18 30)'
const HYBRID_HEX = '#3ac4c4'
const BASELINE_HEX = '#e07c3a'
const SEED_COLORS = ['#3ac4c4', '#5dd9c1', '#22b8a4']

const tooltipStyle = {
  backgroundColor: '#161b2e',
  border: '1px solid #252d3d',
  borderRadius: '8px',
  color: '#e8eaf0',
  fontSize: '12px',
  fontFamily: 'monospace',
}

export default function ChartsSection() {
  const { ref, visible } = useInView()

  return (
    <section id="results" className="py-24 px-4 sm:px-6 max-w-6xl mx-auto" ref={ref}>
      <div className="mb-12">
        <span className="text-xs font-mono text-primary uppercase tracking-widest">Data</span>
        <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
          Experimental Results
        </h2>
        <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
          Charts from the Science Fair 2026 poster. Hybrid S33 consistently outperforms the
          transformer baseline across context lengths, training steps, and random seeds.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" ref={ref}>
        {/* Chart 1: Perplexity vs Context Length */}
        <div
          className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
        >
          <div className="mb-4">
            <p className="text-foreground font-semibold text-sm">
              Perplexity vs Context Length
            </p>
            <p className="text-muted-foreground text-xs mt-1">
              Gap continues to grow — hybrid pulls ahead as context scales
            </p>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={perplexityByCtxLength} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
              <XAxis
                dataKey="ctx"
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#252d3d' }}
                tickLine={false}
                label={{ value: 'Context Length (tokens)', position: 'insideBottom', offset: -2, fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'Perplexity', angle: -90, position: 'insideLeft', fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace', offset: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#6b7a9a' }}
              />
              <Line
                type="monotone"
                dataKey="hybrid"
                name="Hybrid S33"
                stroke={HYBRID_HEX}
                strokeWidth={2.5}
                dot={{ r: 4, fill: HYBRID_HEX, strokeWidth: 0 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey="baseline"
                name="Transformer"
                stroke={BASELINE_HEX}
                strokeWidth={2.5}
                strokeDasharray="5 3"
                dot={{ r: 4, fill: BASELINE_HEX, strokeWidth: 0 }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 2: Training curves */}
        <div
          className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '120ms' }}
        >
          <div className="mb-4">
            <p className="text-foreground font-semibold text-sm">
              Training Perplexity Curves
            </p>
            <p className="text-muted-foreground text-xs mt-1">
              Hybrid descends faster and reaches lower final perplexity
            </p>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trainingCurves} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#252d3d' }}
                tickLine={false}
                label={{ value: 'Training Step', position: 'insideBottom', offset: -2, fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'Perplexity', angle: -90, position: 'insideLeft', fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace', offset: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#6b7a9a' }}
              />
              <Line
                type="monotone"
                dataKey="hybrid"
                name="Hybrid S33"
                stroke={HYBRID_HEX}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="baseline"
                name="Transformer"
                stroke={BASELINE_HEX}
                strokeWidth={2.5}
                strokeDasharray="5 3"
                dot={false}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 3: Seed consistency */}
        <div
          className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '240ms' }}
        >
          <div className="mb-4">
            <p className="text-foreground font-semibold text-sm">
              Seed Consistency — Hybrid S33
            </p>
            <p className="text-muted-foreground text-xs mt-1">
              Three random seeds all converge — results are not due to luck
            </p>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={seedConsistency} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={{ stroke: '#252d3d' }}
                tickLine={false}
                label={{ value: 'Training Step', position: 'insideBottom', offset: -2, fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'Perplexity', angle: -90, position: 'insideLeft', fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace', offset: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend
                wrapperStyle={{ fontSize: 11, fontFamily: 'monospace', color: '#6b7a9a' }}
              />
              {[
                { key: 'seed1', name: 'Seed 1', color: SEED_COLORS[0] },
                { key: 'seed2', name: 'Seed 2', color: SEED_COLORS[1] },
                { key: 'seed3', name: 'Seed 3', color: SEED_COLORS[2] },
              ].map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.name}
                  stroke={s.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Chart 4: Ablation study */}
        <div
          className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
            visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}
          style={{ transitionDelay: '360ms' }}
        >
          <div className="mb-4">
            <p className="text-foreground font-semibold text-sm">
              Ablation Study — Component Contribution
            </p>
            <p className="text-muted-foreground text-xs mt-1">
              Fusion mechanism contributes most to performance gains
            </p>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={ablationData}
              margin={{ top: 5, right: 10, bottom: 30, left: 0 }}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#252d3d" horizontal={false} />
              <XAxis
                type="number"
                domain={[15, 28]}
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                label={{ value: 'Validation Perplexity (lower = better)', position: 'insideBottom', offset: -5, fill: '#6b7a9a', fontSize: 10, fontFamily: 'monospace' }}
              />
              <YAxis
                dataKey="name"
                type="category"
                tick={{ fill: '#6b7a9a', fontSize: 11, fontFamily: 'monospace' }}
                axisLine={false}
                tickLine={false}
                width={90}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(40, 44, 68, 0.5)' }} />
              <ReferenceLine x={18.9} stroke={HYBRID_HEX} strokeDasharray="4 2" strokeOpacity={0.5} />
              <Bar dataKey="ppl" name="Perplexity" radius={[0, 4, 4, 0]}>
                {ablationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Result interpretation */}
      <div
        className={`mt-8 rounded-xl border border-primary/30 bg-primary/5 p-6 transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
        style={{ transitionDelay: '480ms' }}
      >
        <h3 className="text-primary font-bold text-sm mb-3">Result Interpretation</h3>
        <p className="text-muted-foreground text-sm leading-relaxed max-w-3xl">
          The data demonstrates that the split-stream hybrid architecture consistently achieves lower
          perplexity than the transformer baseline across all tested context lengths. The ablation
          experiments further show that the{' '}
          <span className="text-foreground font-medium">fusion of the two modeling streams</span>{' '}
          contributes most strongly to the observed performance gains. This suggests the hybrid
          successfully integrates complementary sequence modeling strategies to produce more accurate
          predictions.
        </p>
      </div>
    </section>
  )
}
