import { Header } from './components/portfolio/Header'
import { Hero } from './components/portfolio/Hero'
import { About } from './components/portfolio/About'
import { Skills } from './components/portfolio/Skills'
import { Experience } from './components/portfolio/Experience'
import { Projects } from './components/portfolio/Projects'
import { Education } from './components/portfolio/Education'
import { Footer } from './components/portfolio/Footer'
import { Chatbot } from './components/chatbot/Chatbot'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <Hero />
        <Chatbot />
        <About />
        <Skills />
        <Experience />
        <Projects />
        <Education />
      </main>
      <Footer />
    </div>
  )
}
