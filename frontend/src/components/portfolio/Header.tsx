export function Header() {
  return (
    <header className="border-b border-zinc-800/80 sticky top-0 z-10 bg-zinc-950/90 backdrop-blur">
      <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <a href="#" className="font-semibold text-zinc-100 hover:text-white transition-colors">
          Developer Name
        </a>
        <nav className="flex items-center gap-6">
          <a href="#experience" className="text-sm text-zinc-400 hover:text-white transition-colors">Experience</a>
          <a href="#projects" className="text-sm text-zinc-400 hover:text-white transition-colors">Projects</a>
        </nav>
      </div>
    </header>
  )
}
