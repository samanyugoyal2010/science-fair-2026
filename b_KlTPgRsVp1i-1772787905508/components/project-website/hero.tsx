'use client'

import { useEffect, useRef } from 'react'

export default function Hero() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animFrame: number
    let particles: { x: number; y: number; vx: number; vy: number; alpha: number; size: number }[] = []

    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    // Spawn particles
    for (let i = 0; i < 60; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        alpha: Math.random() * 0.5 + 0.1,
        size: Math.random() * 2 + 0.5,
      })
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Connect nearby particles
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 120) {
            ctx.beginPath()
            ctx.strokeStyle = `oklch(0.72 0.17 195 / ${0.12 * (1 - dist / 120)})`
            ctx.lineWidth = 0.5
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.stroke()
          }
        }
      }

      // Draw particles
      particles.forEach((p) => {
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fillStyle = `oklch(0.72 0.17 195 / ${p.alpha})`
        ctx.fill()

        p.x += p.vx
        p.y += p.vy

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1
      })

      animFrame = requestAnimationFrame(draw)
    }

    draw()
    return () => {
      cancelAnimationFrame(animFrame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden grid-bg">
      {/* Animated canvas background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        aria-hidden="true"
      />

      {/* Radial glow */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            'radial-gradient(ellipse 70% 50% at 50% 50%, oklch(0.72 0.17 195 / 0.07) 0%, transparent 70%)',
        }}
        aria-hidden="true"
      />

      {/* Content */}
      <div className="relative z-10 text-center px-4 sm:px-6 max-w-4xl mx-auto">
        {/* Eyebrow */}
        <div className="inline-flex items-center gap-2 border border-primary/30 bg-primary/5 text-primary text-xs font-mono px-3 py-1.5 rounded-full mb-6 animate-fade-in-up">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse-glow" />
          Science Fair 2026 &mdash; Research Paper
        </div>

        {/* Main title */}
        <h1
          className="text-4xl sm:text-5xl lg:text-7xl font-bold text-foreground leading-tight text-balance mb-6 animate-fade-in-up"
          style={{ animationDelay: '0.1s' }}
        >
          Split-Stream{' '}
          <span
            className="text-primary"
            style={{
              backgroundImage: 'linear-gradient(90deg, oklch(0.72 0.17 195), oklch(0.65 0.22 160))',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Hybrid
          </span>
          <br />
          Architecture
        </h1>

        <p
          className="text-muted-foreground text-base sm:text-lg leading-relaxed max-w-2xl mx-auto mb-8 text-pretty animate-fade-in-up"
          style={{ animationDelay: '0.2s' }}
        >
          A novel SSM + KAN hybrid architecture that consistently outperforms
          transformer baselines in perplexity while maintaining high memory
          efficiency at small parameter scales.
        </p>

        {/* Stat pills */}
        <div
          className="flex flex-wrap items-center justify-center gap-3 mb-10 animate-fade-in-up"
          style={{ animationDelay: '0.3s' }}
        >
          {[
            { value: 'Lower PPL', label: 'vs Transformer' },
            { value: 'Efficient', label: 'Memory Usage' },
            { value: '3 Seeds', label: 'Validated' },
            { value: '4 Ctx Lengths', label: 'Tested' },
          ].map((stat) => (
            <div
              key={stat.label}
              className="border border-border bg-card/60 backdrop-blur-sm px-4 py-2 rounded-lg flex flex-col items-center"
            >
              <span className="text-primary font-mono font-bold text-sm">{stat.value}</span>
              <span className="text-muted-foreground text-xs">{stat.label}</span>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div
          className="flex flex-wrap items-center justify-center gap-3 animate-fade-in-up"
          style={{ animationDelay: '0.4s' }}
        >
          <a
            href="#results"
            className="bg-primary text-primary-foreground font-semibold text-sm px-5 py-2.5 rounded-lg hover:opacity-90 transition-opacity"
          >
            View Results
          </a>
          <a
            href="#architecture"
            className="border border-border text-foreground font-semibold text-sm px-5 py-2.5 rounded-lg hover:border-primary/60 hover:text-primary transition-colors"
          >
            Explore Architecture
          </a>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 text-muted-foreground">
        <span className="text-xs font-mono">scroll</span>
        <div className="w-px h-8 bg-border relative overflow-hidden">
          <div
            className="absolute top-0 left-0 w-full h-1/2 bg-primary"
            style={{ animation: 'pw-scan-line 1.5s linear infinite' }}
          />
        </div>
      </div>
    </section>
  )
}
