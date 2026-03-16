import { useState, useRef, useEffect } from 'react'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { motion, AnimatePresence } from 'framer-motion'

type Message = { role: 'user' | 'assistant'; content: string }

export function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading])

  const handleSend = async (content: string) => {
    if (!content.trim()) return

    const userMessage: Message = { role: 'user', content }
    setMessages((prev) => [...prev, userMessage])
    setLoading(true)

    try {
      const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || ''
      const response = await fetch(`${apiBaseUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: content }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get an answer from the AI.')
      }

      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer }])
    } catch (error: any) {
      const errorMessage = error.message || 'Network error. Please make sure the backend is running.'
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, I encountered an error: ${errorMessage}` },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="py-12 px-4 border-t border-zinc-800/50 relative">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />

      <div className="max-w-4xl mx-auto">
        <motion.div
          layout
          initial={false}
          animate={{ height: messages.length === 0 ? 240 : 500 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="rounded-3xl border border-zinc-800 bg-zinc-900/40 backdrop-blur-xl overflow-hidden flex flex-col shadow-2xl shadow-blue-500/5 ring-1 ring-white/5"
        >
          <div className="flex-1 overflow-y-auto p-6 space-y-2 custom-scrollbar">
            <AnimatePresence initial={false}>
              {messages.length === 0 ? (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full flex flex-col items-center justify-center text-center px-4"
                >
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-zinc-800 to-zinc-900 flex items-center justify-center mb-4 border border-zinc-700/50 shadow-inner">
                    <span className="text-2xl">✨</span>
                  </div>
                  <div>
                    <h3 className="text-zinc-100 font-semibold text-lg mb-1">Get to know me</h3>
                    <p className="text-zinc-500 text-sm max-w-[280px] leading-relaxed">
                      Ask about my specific skills, professional experience, or previous projects.
                    </p>
                  </div>
                </motion.div>
              ) : (
                messages.map((m, i) => (
                  <ChatMessage key={i} role={m.role} content={m.content} />
                ))
              )}
            </AnimatePresence>

            {loading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-start mb-4"
              >
                <div className="bg-zinc-800/80 rounded-2xl rounded-tl-none px-5 py-3 border border-zinc-700/50">
                  <div className="flex gap-1.5">
                    <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce" />
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <ChatInput onSend={handleSend} disabled={loading} isFirstTime={messages.length === 0} />
        </motion.div>
      </div>
    </section>

  )
}
