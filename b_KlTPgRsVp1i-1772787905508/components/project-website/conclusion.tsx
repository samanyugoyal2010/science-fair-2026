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

const findings = [
  {
    number: '01',
    title: 'Consistent Outperformance',
    body: 'The hybrid architecture achieves lower perplexity than the transformer baseline across all tested context lengths — 128, 256, 512, and 1024 tokens.',
  },
  {
    number: '02',
    title: 'Fusion is the Key Driver',
    body: 'Ablation experiments show the fusion mechanism contributes most strongly to performance gains, confirming the value of combining SSM and KAN streams.',
  },
  {
    number: '03',
    title: 'Statistically Reliable',
    body: 'Multiple random seeds confirm the results are not due to stochastic variation — the hybrid consistently outperforms under identical conditions.',
  },
  {
    number: '04',
    title: 'Hypothesis Confirmed',
    body: 'The split-stream hybrid model improves performance faster relative to a same-parameter transformer while maintaining competitive memory usage.',
  },
]

const references = [
  {
    authors: 'Vaswani, Ashish, et al.',
    title: '"Attention Is All You Need."',
    venue: 'Advances in Neural Information Processing Systems, 2017.',
  },
  {
    authors: 'Gu, Albert, and Tri Dao.',
    title: '"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."',
    venue: 'arXiv preprint, 2023.',
  },
  {
    authors: 'Liu, Ziming, et al.',
    title: '"Kolmogorov–Arnold Networks."',
    venue: 'arXiv preprint, 2024.',
  },
  {
    authors: 'Radford et al.',
    title: 'Language Models are Unsupervised Multitask Learners.',
    venue: 'OpenAI Blog, 2019.',
  },
  {
    authors: 'Brown et al.',
    title: 'Language Models are Few-Shot Learners.',
    venue: 'NeurIPS, 2020.',
  },
]

export default function Conclusion() {
  const { ref, visible } = useInView()

  return (
    <>
      {/* Conclusion section */}
      <section
        id="conclusion"
        className="py-24 px-4 sm:px-6 bg-card/20 border-t border-border"
        ref={ref}
      >
        <div className="max-w-6xl mx-auto">
          <div className="mb-12">
            <span className="text-xs font-mono text-primary uppercase tracking-widest">
              Summary
            </span>
            <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
              Conclusion
            </h2>
            <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
              The results support the hypothesis that hybrid architectures can provide meaningful
              improvements in language modeling performance while maintaining comparable
              computational characteristics.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-12">
            {findings.map((f, i) => (
              <div
                key={f.number}
                className={`border border-border bg-card rounded-xl p-6 transition-all duration-700 ${
                  visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
                }`}
                style={{ transitionDelay: `${i * 100}ms` }}
              >
                <span className="text-xs font-mono text-muted-foreground block mb-3">
                  {f.number}
                </span>
                <h3 className="text-foreground font-bold text-sm mb-2">{f.title}</h3>
                <p className="text-muted-foreground text-xs leading-relaxed">{f.body}</p>
              </div>
            ))}
          </div>

          {/* Final callout */}
          <div
            className={`rounded-2xl border border-primary/40 bg-primary/5 p-8 text-center transition-all duration-700 ${
              visible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
            }`}
            style={{ transitionDelay: '500ms' }}
          >
            <p className="text-xs font-mono text-primary uppercase tracking-widest mb-3">
              Key Takeaway
            </p>
            <p className="text-foreground text-lg sm:text-xl font-medium max-w-2xl mx-auto leading-relaxed text-balance">
              Combining attention-based sequence modeling with state-space modeling can significantly
              improve language model prediction accuracy at small parameter scales.
            </p>
            <p className="text-muted-foreground text-sm mt-4 max-w-xl mx-auto leading-relaxed">
              This suggests hybrid architectures may democratize access to capable AI systems in
              environments with limited computational resources.
            </p>
          </div>
        </div>
      </section>

      {/* References section */}
      <section className="py-16 px-4 sm:px-6 border-t border-border">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <span className="text-xs font-mono text-primary uppercase tracking-widest">
              Citations
            </span>
            <h2 className="text-2xl font-bold text-foreground mt-2">Referenced Work</h2>
          </div>

          <div className="space-y-3">
            {references.map((ref, i) => (
              <div
                key={i}
                className="flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-4 border border-border bg-card rounded-lg px-5 py-4"
              >
                <span className="text-xs font-mono text-primary w-5 flex-shrink-0 mt-0.5">
                  [{i + 1}]
                </span>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  <span className="text-foreground font-medium">{ref.authors} </span>
                  <em>{ref.title}</em>{' '}
                  <span className="text-muted-foreground">{ref.venue}</span>
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-4 sm:px-6">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="w-6 h-6 rounded bg-primary flex items-center justify-center text-primary-foreground font-mono text-[10px] font-bold">
              S33
            </span>
            <span className="text-muted-foreground text-xs">Split-Stream Hybrid Architecture</span>
          </div>
          <p className="text-muted-foreground text-xs font-mono">Science Fair 2026</p>
        </div>
      </footer>
    </>
  )
}
