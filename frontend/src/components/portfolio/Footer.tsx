import { BackendStatus } from '../BackendStatus'

export function Footer() {
  return (
    <footer className="border-t border-zinc-800/80 py-8 px-4 mt-auto">
      <div className="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-zinc-500">
        <div>© {new Date().getFullYear()} Developer Name. All rights reserved.</div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900/50 border border-zinc-800" title="System Status">
          <span className="text-xs uppercase tracking-wider text-zinc-600">Backend</span>
          <BackendStatus />
        </div>
      </div>
    </footer>
  )
}
