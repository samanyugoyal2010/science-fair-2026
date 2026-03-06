'use client'

import { useState, useEffect } from 'react'

const navLinks = [
  { label: 'Problem', href: '#problem' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Math', href: '#math' },
  { label: 'Comparison', href: '#comparison' },
  { label: 'Results', href: '#results' },
  { label: 'Conclusion', href: '#conclusion' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-background/90 backdrop-blur-md border-b border-border'
          : 'bg-transparent'
      }`}
    >
      <nav className="max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2 group">
          <span className="w-7 h-7 rounded-md bg-primary/15 border border-primary/40 flex items-center justify-center">
            <span className="block w-2 h-2 rounded-full bg-primary animate-pulse" />
          </span>
          <span className="text-foreground font-semibold text-sm hidden sm:block tracking-wide">
            Split-Stream Hybrid
          </span>
        </a>

        {/* Desktop links */}
        <ul className="hidden md:flex items-center gap-6">
          {navLinks.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                className="text-muted-foreground hover:text-primary text-sm transition-colors duration-200"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>

        {/* Science Fair Badge */}
        <a
          href="#conclusion"
          className="hidden md:flex items-center gap-2 text-xs font-mono border border-primary/40 text-primary px-3 py-1.5 rounded-full hover:bg-primary/10 transition-colors"
        >
          Science Fair 2026
        </a>

        {/* Mobile hamburger */}
        <button
          className="md:hidden flex flex-col gap-1.5 p-2"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span
            className={`block h-0.5 w-5 bg-foreground transition-transform duration-200 ${menuOpen ? 'rotate-45 translate-y-2' : ''}`}
          />
          <span
            className={`block h-0.5 w-5 bg-foreground transition-opacity duration-200 ${menuOpen ? 'opacity-0' : ''}`}
          />
          <span
            className={`block h-0.5 w-5 bg-foreground transition-transform duration-200 ${menuOpen ? '-rotate-45 -translate-y-2' : ''}`}
          />
        </button>
      </nav>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden bg-card border-b border-border px-4 pb-4">
          <ul className="flex flex-col gap-3">
            {navLinks.map((link) => (
              <li key={link.href}>
                <a
                  href={link.href}
                  onClick={() => setMenuOpen(false)}
                  className="text-muted-foreground hover:text-primary text-sm block py-1 transition-colors"
                >
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </header>
  )
}
