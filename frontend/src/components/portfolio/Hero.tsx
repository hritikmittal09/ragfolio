import { motion } from 'framer-motion'

export function Hero() {
  return (
    <section className="py-24 px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="max-w-4xl mx-auto text-center"
      >
        <h1 className="text-5xl sm:text-7xl font-bold text-white mb-6 tracking-tight">
          Hi, I'm <span className="text-blue-500">Developer Name</span>
        </h1>
        <p className="text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed">
          Full-stack developer specializing in building exceptional digital experiences.
          Currently focused on creating scalable, user-centric applications.
        </p>
      </motion.div>
    </section>
  )
}
