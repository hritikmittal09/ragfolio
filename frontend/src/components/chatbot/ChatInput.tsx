import { useState } from 'react'
import type { FormEvent } from 'react'
import { motion } from 'framer-motion'

interface ChatInputProps {
  onSend: (message: string) => void
  disabled?: boolean
  isFirstTime?: boolean
}

export function ChatInput({ onSend, disabled, isFirstTime }: ChatInputProps) {
  const [value, setValue] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = value.trim()
    if (trimmed && !disabled) {
      onSend(trimmed)
      setValue('')
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="p-4 bg-zinc-950/50 backdrop-blur-md border-t border-zinc-800/50 flex gap-3 relative group"
    >
      <motion.div
        className="flex-1 relative"
        animate={isFirstTime ? {
          boxShadow: [
            "0 0 0px 0px rgba(59, 130, 246, 0)",
            "0 0 20px 2px rgba(59, 130, 246, 0.3)",
            "0 0 0px 0px rgba(59, 130, 246, 0)"
          ]
        } : {}}
        transition={isFirstTime ? {
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut"
        } : {}}
        style={{ borderRadius: '12px' }}
      >
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Ask anything about the portfolio..."
          disabled={disabled}
          className="w-full bg-zinc-900/80 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all"
        />
      </motion.div>
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        type="submit"
        disabled={disabled || !value.trim()}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-800 disabled:text-zinc-500 text-white rounded-xl px-5 py-3 text-sm font-semibold transition-colors shadow-lg shadow-blue-500/10"
      >
        Send
      </motion.button>
    </form>
  )
}
