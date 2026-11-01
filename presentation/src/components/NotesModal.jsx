import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function NotesModal({ isOpen, onClose, currentSlide, totalSlides, title, notes }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  // Split notes into paragraphs
  const paragraphs = notes ? notes.split('\n\n').filter(p => p.trim().length > 0) : [];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8 bg-slate-900/60 backdrop-blur-xs">
        {/* Backdrop clickable */}
        <div className="absolute inset-0" onClick={onClose} />

        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 15 }}
          transition={{ duration: 0.2 }}
          className="relative bg-white rounded-2xl shadow-2xl border border-slate-200 w-full max-w-3xl max-h-[85vh] flex flex-col z-10 overflow-hidden"
        >
          {/* Header */}
          <div className="px-6 py-4 bg-gradient-to-r from-amber-50 to-orange-50 border-b border-amber-200/80 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="p-2 bg-amber-500 text-white rounded-lg shadow-xs">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z" />
                </svg>
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-amber-700 bg-amber-100 px-2 py-0.5 rounded-full border border-amber-300">
                    Mode Orateur • Slide {currentSlide + 1}/{totalSlides}
                  </span>
                  <span className="text-xs text-slate-500 font-mono">Appuyez sur 'N' ou 'Échap' pour fermer</span>
                </div>
                <h3 className="text-base font-bold text-slate-900 line-clamp-1 mt-0.5">
                  {title}
                </h3>
              </div>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
              title="Fermer (Échap)"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Body content with scroll */}
          <div className="p-6 md:p-8 overflow-y-auto space-y-4 text-slate-700 text-sm md:text-base leading-relaxed">
            {paragraphs.map((p, idx) => (
              <p
                key={idx}
                className={`p-3.5 rounded-xl transition-all ${
                  idx === 0
                    ? 'bg-blue-50/70 border-l-4 border-blue-600 font-medium text-slate-800'
                    : idx === paragraphs.length - 1
                    ? 'bg-emerald-50/70 border-l-4 border-emerald-600 font-medium text-slate-800'
                    : 'bg-slate-50 border border-slate-100 hover:bg-slate-100/60'
                }`}
              >
                {p}
              </p>
            ))}
          </div>

          {/* Footer */}
          <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between text-xs text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              Discours académique calibré pour une soutenance de 15 à 20 minutes
            </span>
            <button
              onClick={onClose}
              className="px-3 py-1 bg-slate-200 hover:bg-slate-300 text-slate-800 font-semibold rounded-md transition-colors"
            >
              Compris
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
