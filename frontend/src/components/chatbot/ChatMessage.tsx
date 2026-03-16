import { motion } from 'framer-motion'

type Role = 'user' | 'assistant'

interface ChatMessageProps {
  role: Role
  content: string
}

export function ChatMessage({ role, content }: ChatMessageProps) {
  const isUser = role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
    >
      <div
        className={`max-w-[80%] rounded-2xl px-5 py-3 shadow-lg ${isUser
            ? 'bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-tr-none'
            : 'bg-zinc-800/80 text-zinc-100 border border-zinc-700/50 backdrop-blur-sm rounded-tl-none'
          }`}
      >
        <div className="flex flex-col gap-1">
          <span className={`text-[10px] uppercase tracking-widest font-bold opacity-50 ${isUser ? 'text-right' : 'text-left'}`}>
            {isUser ? 'You' : 'Assistant'}
          </span>
          <p className="text-[14.5px] leading-relaxed whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    </motion.div>
  )
}
