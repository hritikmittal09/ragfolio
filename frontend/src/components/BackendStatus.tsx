import { useEffect, useState } from 'react'

const POLL_MS = 5000

export function BackendStatus() {
  const [alive, setAlive] = useState<boolean | null>(null)

  useEffect(() => {
    const check = async () => {
      try {
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '/api' : '')
        const res = await fetch(`${apiBaseUrl}/health`)
        setAlive(res.ok)
      } catch {
        setAlive(false)
      }
    }
    check()
    const id = setInterval(check, POLL_MS)
    return () => clearInterval(id)
  }, [])

  return (
    <span
      className="inline-flex items-center gap-1.5 cursor-default"
      title={alive === true ? 'Backend alive' : alive === false ? 'Backend offline' : 'Checking…'}
    >
      <span
        className={`h-2 w-2 rounded-full shrink-0 ${alive === true ? 'bg-emerald-500' : alive === false ? 'bg-red-500' : 'bg-zinc-500 animate-pulse'
          }`}
        aria-hidden
      />
      <span className="text-xs text-zinc-400 sr-only sm:not-sr-only">
        {alive === true ? 'Backend alive' : alive === false ? 'Backend offline' : 'Checking…'}
      </span>
    </span>
  )
}
