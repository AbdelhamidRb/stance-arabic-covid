import React from 'react';

export default function NavBar({
  currentSlide,
  totalSlides,
  onPrev,
  onNext,
  onToggleNotes,
  showNotes,
  slides,
  onOpenPresenter
}) {
  const current = slides[currentSlide];
  const progressPercent = ((currentSlide + 1) / totalSlides) * 100;

  return (
    <nav className="h-16 bg-white border-t border-slate-200 px-6 flex items-center justify-between shadow-sm relative z-20 flex-shrink-0 select-none">
      {/* Top micro progress line */}
      <div className="absolute top-0 left-0 right-0 h-1 bg-slate-100">
        <div
          className="h-full bg-gradient-to-r from-blue-600 via-sky-500 to-indigo-600 transition-all duration-300 ease-out"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Left: Slide title & counter */}
      <div className="flex items-center gap-4 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-slate-800 bg-slate-100 px-2.5 py-1 rounded-md border border-slate-200 font-mono">
            {String(currentSlide + 1).padStart(2, '0')} / {String(totalSlides).padStart(2, '0')}
          </span>
        </div>
        <div className="hidden sm:block truncate max-w-xs md:max-w-md lg:max-w-xl">
          <span className="text-sm font-semibold text-slate-800 truncate block">
            {current?.title || `Slide ${currentSlide + 1}`}
          </span>
        </div>
      </div>

      {/* Center: Slide dots navigation */}
      <div className="hidden md:flex items-center gap-1.5 px-3 py-1 bg-slate-50 rounded-full border border-slate-200/80">
        {slides.map((s, idx) => (
          <button
            key={s.id}
            onClick={() => {
              // Could navigate directly
            }}
            title={`Slide ${idx + 1}: ${s.title}`}
            className={`transition-all duration-200 rounded-full ${
              idx === currentSlide
                ? 'w-6 h-2.5 bg-blue-600 rounded-full shadow-sm'
                : 'w-2 h-2 bg-slate-300 hover:bg-slate-400'
            }`}
          />
        ))}
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2.5">
        {/* Double Screen Presenter Window Button */}
        <button
          onClick={onOpenPresenter}
          className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-100 hover:bg-slate-800 border border-slate-700 shadow-xs transition-colors"
          title="Ouvrir la console orateur sur votre écran PC (Datashow reste propre)"
        >
          <span className="text-sm">🖥️</span>
          <span>Double Écran (Orateur)</span>
        </button>

        {/* In-slide Notes Modal Toggle Button */}
        <button
          onClick={onToggleNotes}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
            showNotes
              ? 'bg-amber-500 text-white border-amber-600 shadow-sm'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border-slate-300'
          }`}
          title="Afficher/masquer les notes orateur en pop-up (Touche N)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="hidden sm:inline">Notes</span>
          <kbd className="px-1.5 py-0.5 text-[10px] bg-white/30 text-current rounded border border-current/30 font-mono">
            N
          </kbd>
        </button>

        {/* Prev Button */}
        <button
          onClick={onPrev}
          disabled={currentSlide === 0}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-bold bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 disabled:opacity-35 disabled:cursor-not-allowed shadow-xs transition-colors"
          title="Slide précédente (Flèche gauche)"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span className="hidden sm:inline">Précédent</span>
        </button>

        {/* Next Button */}
        <button
          onClick={onNext}
          disabled={currentSlide === totalSlides - 1}
          className="flex items-center gap-1 px-3.5 py-1.5 rounded-lg text-xs font-bold bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-35 disabled:cursor-not-allowed shadow-sm transition-colors"
          title="Slide suivante (Flèche droite ou Espace)"
        >
          <span>Suivant</span>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </nav>
  );
}
