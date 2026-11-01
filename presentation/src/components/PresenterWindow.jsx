import React, { useState, useEffect } from 'react';
import { slides } from '../data/slides';

export default function PresenterWindow({ currentSlide, totalSlides, onGoToSlide }) {
  const [seconds, setSeconds] = useState(0);
  const [isTimerRunning, setIsTimerRunning] = useState(true);

  // Timer logic for defense rehearsal
  useEffect(() => {
    let interval = null;
    if (isTimerRunning) {
      interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isTimerRunning]);

  const formatTime = (totalSec) => {
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const current = slides[currentSlide];
  const next = currentSlide < totalSlides - 1 ? slides[currentSlide + 1] : null;
  const notesParagraphs = current?.notes ? current.notes.split('\n\n').filter(p => p.trim().length > 0) : [];

  return (
    <div className="h-screen w-screen bg-slate-950 text-slate-100 flex flex-col font-sans select-none overflow-hidden">
      {/* Top Header: Presenter Status & Timer */}
      <header className="h-14 bg-slate-900 border-b border-slate-800 px-6 flex items-center justify-between flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span className="text-sm font-bold text-white tracking-wide uppercase">
            Console Orateur M2 (Écran Privé)
          </span>
          <span className="text-xs font-mono bg-emerald-950 text-emerald-400 border border-emerald-800 px-2.5 py-0.5 rounded-md font-semibold">
            🟢 Synchronisé avec le Datashow
          </span>
        </div>

        {/* Center/Right: Timer & Slide counter */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1 rounded-lg">
            <span className="text-xs text-slate-400">⏱️ Durée :</span>
            <span className={`text-base font-mono font-bold ${seconds > 1200 ? 'text-rose-400 animate-pulse' : 'text-emerald-400'}`}>
              {formatTime(seconds)}
            </span>
            <button
              onClick={() => setIsTimerRunning(prev => !prev)}
              className="text-xs text-slate-400 hover:text-white px-1.5 py-0.5 bg-slate-800 rounded ml-1 transition-colors"
              title={isTimerRunning ? 'Mettre en pause le chrono' : 'Reprendre le chrono'}
            >
              {isTimerRunning ? '⏸' : '▶'}
            </button>
            <button
              onClick={() => setSeconds(0)}
              className="text-xs text-slate-400 hover:text-white px-1.5 py-0.5 bg-slate-800 rounded transition-colors"
              title="Réinitialiser le chrono à 00:00"
            >
              ↺
            </button>
          </div>

          <div className="text-sm font-bold text-slate-200 bg-slate-800 border border-slate-700 px-3 py-1 rounded-lg font-mono">
            Slide {currentSlide + 1} / {totalSlides}
          </div>
        </div>
      </header>

      {/* Main Body */}
      <div className="flex-1 grid grid-cols-12 min-h-0 divide-x divide-slate-800 overflow-hidden">
        {/* Left column (4 cols): Slide previews & quick access */}
        <div className="col-span-4 p-5 flex flex-col justify-between overflow-y-auto space-y-4 bg-slate-900/50">
          {/* Current slide preview */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-sm">
            <div className="text-[11px] font-bold text-blue-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <span>📽️</span> Actuellement affiché au Jury (Datashow)
            </div>
            <h3 className="text-base font-bold text-white">
              {current?.title}
            </h3>
            {current?.subtitle && (
              <p className="text-xs text-slate-400 mt-1 leading-snug">{current.subtitle}</p>
            )}
          </div>

          {/* Next slide preview */}
          <div className="bg-slate-950/70 p-4 rounded-xl border border-dashed border-slate-800">
            <div className="text-[11px] font-bold text-amber-400 uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <span>⏭️</span> Diapositive suivante (À anticiper)
            </div>
            {next ? (
              <div>
                <h4 className="text-sm font-bold text-slate-200">{next.title}</h4>
                <p className="text-xs text-slate-500 line-clamp-2 mt-0.5">{next.subtitle}</p>
              </div>
            ) : (
              <div className="text-xs text-slate-500 italic">Dernière diapositive de la soutenance</div>
            )}
          </div>

          {/* Quick jump to slides 1-12 */}
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800">
            <div className="text-[11px] text-slate-400 font-bold uppercase mb-2 flex items-center justify-between">
              <span>Accès direct aux 12 diapositives</span>
              <span className="text-[10px] text-slate-500 font-mono">Clic = saut direct</span>
            </div>
            <div className="grid grid-cols-6 gap-1.5">
              {slides.map((s, idx) => (
                <button
                  key={s.id}
                  onClick={() => onGoToSlide(idx)}
                  className={`py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
                    idx === currentSlide
                      ? 'bg-blue-600 text-white shadow-sm ring-1 ring-blue-400'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                  }`}
                  title={`${idx + 1}. ${s.title}`}
                >
                  {String(idx + 1).padStart(2, '0')}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right column (8 cols): Speech notes to read */}
        <div className="col-span-8 p-6 flex flex-col justify-between overflow-hidden bg-slate-950">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <span className="text-xs font-bold text-amber-400 uppercase tracking-wider block">
                Ce que vous devez dire (Notes Orateur)
              </span>
              <h2 className="text-lg font-bold text-white line-clamp-1 mt-0.5">
                Slide {currentSlide + 1} : {current?.title}
              </h2>
            </div>
            {next && (
              <span className="text-xs text-slate-500">
                Suivante : <strong className="text-slate-400">{next.title.slice(0, 25)}...</strong>
              </span>
            )}
          </div>

          {/* Scrollable Speech Notes */}
          <div className="flex-1 overflow-y-auto pr-2 py-4 space-y-3 min-h-0 select-text">
            {notesParagraphs.map((para, i) => (
              <p
                key={i}
                className={`p-4 rounded-xl text-base leading-relaxed ${
                  i === 0
                    ? 'bg-blue-950/35 border-l-4 border-blue-500 text-blue-100 font-medium'
                    : i === notesParagraphs.length - 1
                    ? 'bg-emerald-950/35 border-l-4 border-emerald-500 text-emerald-100 font-medium'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200'
                }`}
              >
                {para}
              </p>
            ))}
          </div>
        </div>
      </div>

      {/* Sticky Bottom Bar with big controls */}
      <footer className="h-16 bg-slate-900 border-t border-slate-800 px-6 flex items-center justify-between gap-4 flex-shrink-0 z-20">
        <button
          onClick={() => onGoToSlide(Math.max(0, currentSlide - 1))}
          disabled={currentSlide === 0}
          className="flex-1 py-2.5 bg-slate-800 hover:bg-slate-700 active:bg-slate-600 disabled:opacity-30 rounded-xl text-sm font-bold text-slate-200 transition-colors shadow-xs"
        >
          ← Diapositive Précédente
        </button>

        <span className="font-mono text-sm font-bold text-slate-300 px-4">
          {currentSlide + 1} / {totalSlides}
        </span>

        <button
          onClick={() => onGoToSlide(Math.min(totalSlides - 1, currentSlide + 1))}
          disabled={currentSlide === totalSlides - 1}
          className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-30 rounded-xl text-sm font-bold text-white transition-colors shadow-sm"
        >
          Diapositive Suivante →
        </button>
      </footer>
    </div>
  );
}
