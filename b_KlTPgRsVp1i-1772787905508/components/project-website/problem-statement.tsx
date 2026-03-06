'use client'

import { useEffect, useRef, useState } from 'react'

const cards = [
  {
    number: '01',
    title: 'Problem Statement',
    body: 'Modern AI relies heavily on transformer architectures requiring significant computational resources. As models scale, systems with limited memory struggle to deploy high-quality models. This project investigates a hybrid architecture that improves performance while maintaining efficiency.',
    accent: false,
  },
  {
    number: '02',
    title: 'Research Questions',
    body: null,
    bullets: [
      'Can a hybrid architecture reduce peak GPU memory vs. a same-parameter transformer?',
      'Can memory reduction be achieved without degrading performance — rather, improving it?',
    ],
    accent: false,
  },
  {
    number: '03',
    title: 'Hypothesis',
    body: 'If a split-stream hybrid model combining a state-space sequence model (SSM) and a Kolmogorov-Arnold Network (KAN) is used, it will improve performance faster relative to a same-parameter transformer while maintaining or reducing memory usage.',
    accent: true,
  },
]

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

export default function ProblemStatement() {
  const { ref, visible } = useInView()

  return (
    <section id="problem" className="py-24 px-4 sm:px-6 max-w-6xl mx-auto" ref={ref}>
      {/* Section label */}
      <div className="mb-12">
        <span className="text-xs font-mono text-primary uppercase tracking-widest">Background</span>
        <h2 className="text-3xl sm:text-4xl font-bold text-foreground mt-2 text-balance">
          The Problem We Solve
        </h2>
        <p className="text-muted-foreground mt-3 max-w-xl text-pretty leading-relaxed">
          Over 212,000 AI companies rely on transformer architectures. Democratizing access requires
          efficient alternatives at small parameter scales.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {cards.map((card, i) => (
          <div
            key={card.number}
            className={`rounded-xl border p-6 transition-all duration-700 ${
              visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
            } ${
              card.accent
                ? 'border-primary/40 bg-primary/5 col-span-1'
                : 'border-border bg-card'
            }`}
            style={{ transitionDelay: `${i * 120}ms` }}
          >
            <span className="text-xs font-mono text-muted-foreground mb-4 block">{card.number}</span>
            <h3
              className={`font-bold text-lg mb-3 ${
                card.accent ? 'text-primary' : 'text-foreground'
              }`}
            >
              {card.title}
            </h3>
            {card.body && (
              <p className="text-muted-foreground text-sm leading-relaxed">{card.body}</p>
            )}
            {card.bullets && (
              <ul className="space-y-3 mt-1">
                {card.bullets.map((b) => (
                  <li key={b} className="flex items-start gap-2 text-sm text-muted-foreground">
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                    {b}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>

      {/* Key idea callout */}
      <div
        className={`mt-6 rounded-xl border border-border bg-card p-6 transition-all duration-700 ${
          visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
        style={{ transitionDelay: '400ms' }}
      >
        <div className="flex flex-col sm:flex-row gap-4 items-start">
          <div className="flex-shrink-0 border border-primary/30 bg-primary/10 text-primary text-xs font-mono px-3 py-1 rounded-full">
            Key Idea
          </div>
          <p className="text-muted-foreground text-sm leading-relaxed">
            Instead of using a single processing pathway like a transformer, the proposed architecture
            processes sequences through{' '}
            <span className="text-foreground font-medium">two parallel streams</span> and combines
            their representations through a fusion mechanism. This hybrid approach leverages the
            strengths of both attention-based and state-space modeling, enabling more capable AI in
            environments with limited computational resources.
          </p>
        </div>
      </div>
    </section>
  )
}
