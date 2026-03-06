import Navbar from '@/components/project-website/navbar'
import Hero from '@/components/project-website/hero'
import ProblemStatement from '@/components/project-website/problem-statement'
import Architecture from '@/components/project-website/architecture'
import MathSection from '@/components/project-website/math-section'
import SpeedComparison from '@/components/project-website/speed-comparison'
import ChartsSection from '@/components/project-website/charts-section'
import Conclusion from '@/components/project-website/conclusion'

export default function ProjectWebsitePage() {
  return (
    <main className="min-h-screen bg-background">
      <Navbar />
      <Hero />
      <ProblemStatement />
      <Architecture />
      <MathSection />
      <SpeedComparison />
      <ChartsSection />
      <Conclusion />
    </main>
  )
}
