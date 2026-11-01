import React from 'react';
import { motion } from 'framer-motion';

const variants = {
  enter: (dir) => ({
    x: dir > 0 ? 60 : -60,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
    transition: {
      x: { type: "spring", stiffness: 350, damping: 35 },
      opacity: { duration: 0.2 }
    }
  },
  exit: (dir) => ({
    x: dir > 0 ? -60 : 60,
    opacity: 0,
    transition: {
      x: { type: "spring", stiffness: 350, damping: 35 },
      opacity: { duration: 0.15 }
    }
  })
};

export default function SlideWrapper({ children, direction = 1, slideNumber, totalSlides, title, subtitle }) {
  return (
    <motion.div
      custom={direction}
      variants={variants}
      initial="enter"
      animate="center"
      exit="exit"
      className="w-full h-full flex flex-col justify-between overflow-y-auto px-8 py-5 md:px-10 md:py-6 bg-white"
    >
      {/* Header section: Sober, Clean, Academic */}
      <div className="mb-3 pb-2.5 border-b border-slate-200 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 flex-shrink-0">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[11px] font-bold tracking-wider uppercase text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded">
              Slide {String(slideNumber).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
            </span>
            <span className="text-xs text-slate-400 font-medium">
              Soutenance M2 • Stance Detection en Arabe
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            {title}
          </h1>
          {subtitle && (
            <p className="text-xs md:text-sm text-slate-500 font-medium mt-0.5">
              {subtitle}
            </p>
          )}
        </div>

        <div className="hidden md:block text-right text-xs text-slate-400 font-mono">
          Corpus Mawqif (Twitter)
        </div>
      </div>

      {/* Main Slide Content Area */}
      <div className="flex-1 flex flex-col justify-center min-h-0">
        {children}
      </div>
    </motion.div>
  );
}
