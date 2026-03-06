import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import 'katex/dist/katex.min.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
})

export const metadata: Metadata = {
  title: 'Split-Stream Hybrid Architecture — S33 Research',
  description:
    'A science fair research project investigating a hybrid SSM+KAN architecture that outperforms transformer baselines at small parameter scales with high memory efficiency.',
  keywords: ['AI', 'machine learning', 'hybrid architecture', 'SSM', 'KAN', 'transformer'],
}

export default function ProjectWebsiteLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div
      className={`pw-theme ${inter.variable} ${jetbrainsMono.variable}`}
      style={{ fontFamily: 'var(--font-inter), sans-serif' }}
    >
      {children}
    </div>
  )
}
