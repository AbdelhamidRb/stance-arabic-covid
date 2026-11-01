import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { AnimatePresence } from 'framer-motion';
import { slides } from './data/slides';
import NavBar from './components/NavBar';
import NotesModal from './components/NotesModal';
import SlideWrapper from './components/SlideWrapper';
import PresenterWindow from './components/PresenterWindow';

// Slide components
import Slide01_Intro from './slides/Slide01_Intro';
import Slide02_StanceVsSentiment from './slides/Slide02_StanceVsSentiment';
import Slide03_Data from './slides/Slide03_Data';
import Slide04_Metrics from './slides/Slide04_Metrics';
import Slide05_Baselines from './slides/Slide05_Baselines';
import Slide06_NLI from './slides/Slide06_NLI';
import Slide07_FromScratch from './slides/Slide07_FromScratch';
import Slide08_PLMs from './slides/Slide08_PLMs';
import Slide09_MultiTask from './slides/Slide09_MultiTask';
import Slide10_TestRobustesse from './slides/Slide10_TestRobustesse';
import Slide11_Errors from './slides/Slide11_Errors';
import Slide12_Production from './slides/Slide12_Production';

const SLIDE_COMPONENTS = [
  Slide01_Intro,
  Slide02_StanceVsSentiment,
  Slide03_Data,
  Slide04_Metrics,
  Slide05_Baselines,
  Slide06_NLI,
  Slide07_FromScratch,
  Slide08_PLMs,
  Slide09_MultiTask,
  Slide10_TestRobustesse,
  Slide11_Errors,
  Slide12_Production
];

export default function App() {
  const isPresenterMode = useMemo(() => {
    return typeof window !== 'undefined' && window.location.search.includes('presenter=true');
  }, []);

  const [currentSlide, setCurrentSlide] = useState(0);
  const [direction, setDirection] = useState(1);
  const [showNotes, setShowNotes] = useState(false);

  const totalSlides = slides.length;

  // Inter-window real-time synchronization via BroadcastChannel & LocalStorage fallback
  useEffect(() => {
    let channel = null;
    try {
      channel = new BroadcastChannel('stance_presentation_channel');
      channel.onmessage = (event) => {
        if (event.data?.type === 'CHANGE_SLIDE' && typeof event.data.index === 'number') {
          setCurrentSlide(event.data.index);
          setDirection(event.data.direction || 1);
        }
      };
    } catch {
      // Fallback
    }

    const handleStorage = (e) => {
      if (e.key === 'stance_slide_index' && e.newValue) {
        const parsed = JSON.parse(e.newValue);
        setCurrentSlide(parsed.index);
        setDirection(parsed.direction || 1);
      }
    };
    window.addEventListener('storage', handleStorage);

    return () => {
      if (channel) channel.close();
      window.removeEventListener('storage', handleStorage);
    };
  }, []);

  const broadcastSlide = useCallback((newIndex, newDir) => {
    try {
      const channel = new BroadcastChannel('stance_presentation_channel');
      channel.postMessage({ type: 'CHANGE_SLIDE', index: newIndex, direction: newDir });
      channel.close();
    } catch {
      // Ignore
    }
    try {
      localStorage.setItem('stance_slide_index', JSON.stringify({ index: newIndex, direction: newDir, timestamp: Date.now() }));
    } catch {
      // Ignore
    }
  }, []);

  const goToNext = useCallback(() => {
    if (currentSlide < totalSlides - 1) {
      const nextIdx = currentSlide + 1;
      setDirection(1);
      setCurrentSlide(nextIdx);
      broadcastSlide(nextIdx, 1);
    }
  }, [currentSlide, totalSlides, broadcastSlide]);

  const goToPrev = useCallback(() => {
    if (currentSlide > 0) {
      const prevIdx = currentSlide - 1;
      setDirection(-1);
      setCurrentSlide(prevIdx);
      broadcastSlide(prevIdx, -1);
    }
  }, [currentSlide, broadcastSlide]);

  const goToSlide = useCallback((idx) => {
    const dir = idx >= currentSlide ? 1 : -1;
    setDirection(dir);
    setCurrentSlide(idx);
    broadcastSlide(idx, dir);
  }, [currentSlide, broadcastSlide]);

  const toggleNotes = useCallback(() => {
    setShowNotes((prev) => !prev);
  }, []);

  const openPresenterWindow = useCallback(() => {
    window.open(
      `${window.location.origin}/?presenter=true`,
      'PresenterConsole',
      'width=1200,height=800,menubar=no,toolbar=no,location=no,status=no'
    );
  }, []);

  // Keyboard Navigation Listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      switch (e.key) {
        case 'ArrowRight':
        case 'PageDown':
        case ' ':
          e.preventDefault();
          goToNext();
          break;
        case 'ArrowLeft':
        case 'PageUp':
          e.preventDefault();
          goToPrev();
          break;
        case 'n':
        case 'N':
          if (!isPresenterMode) {
            e.preventDefault();
            toggleNotes();
          }
          break;
        case 'p':
        case 'P':
          if (!isPresenterMode) {
            e.preventDefault();
            openPresenterWindow();
          }
          break;
        case 'Escape':
          if (showNotes) {
            e.preventDefault();
            setShowNotes(false);
          }
          break;
        case 'Home':
          e.preventDefault();
          goToSlide(0);
          break;
        case 'End':
          e.preventDefault();
          goToSlide(totalSlides - 1);
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToNext, goToPrev, toggleNotes, showNotes, totalSlides, goToSlide, isPresenterMode, openPresenterWindow]);

  // If this window is the Dedicated Presenter Console
  if (isPresenterMode) {
    return (
      <PresenterWindow
        currentSlide={currentSlide}
        totalSlides={totalSlides}
        onGoToSlide={goToSlide}
      />
    );
  }

  // Otherwise: Main Presentation Window (Clean for Datashow Projector)
  const CurrentSlideComponent = SLIDE_COMPONENTS[currentSlide] || Slide01_Intro;
  const currentSlideData = slides[currentSlide];

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-100 text-slate-900 select-none overflow-hidden font-sans">
      {/* Top micro brand line */}
      <header className="h-8 bg-slate-900 text-slate-300 px-6 flex items-center justify-between text-xs border-b border-slate-800 flex-shrink-0 z-10">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 font-bold text-white tracking-wide">
            <span className="text-blue-400">◆</span> Soutenance M2 IA & TALN
          </span>
          <span className="text-slate-500">•</span>
          <span className="text-slate-400 truncate">
            Détection de Position en Arabe (Corpus Mawqif)
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px] text-slate-400 font-mono">
          <span className="hidden sm:inline">Raccourcis : [←] [→] Navigation • [P] Double Écran • [N] Notes</span>
          <button
            onClick={openPresenterWindow}
            className="text-emerald-400 hover:text-emerald-300 font-bold transition-colors flex items-center gap-1"
            title="Ouvre une fenêtre orateur séparée pour votre écran PC"
          >
            <span>🖥️ Mode Double Écran (P)</span>
          </button>
        </div>
      </header>

      {/* Main Slide Presentation Stage (16:9 responsive frame) */}
      <main className="flex-1 flex items-center justify-center p-2 sm:p-4 md:p-6 overflow-hidden">
        <div className="w-full h-full max-w-7xl bg-white rounded-2xl shadow-xl border border-slate-200/90 flex flex-col overflow-hidden relative">
          <AnimatePresence mode="wait" custom={direction}>
            <SlideWrapper
              key={currentSlide}
              direction={direction}
              slideNumber={currentSlide + 1}
              totalSlides={totalSlides}
              title={currentSlideData.title}
              subtitle={currentSlideData.subtitle}
            >
              <CurrentSlideComponent />
            </SlideWrapper>
          </AnimatePresence>
        </div>
      </main>

      {/* Fixed Bottom Navigation Bar */}
      <NavBar
        currentSlide={currentSlide}
        totalSlides={totalSlides}
        onPrev={goToPrev}
        onNext={goToNext}
        onToggleNotes={toggleNotes}
        showNotes={showNotes}
        slides={slides}
        onOpenPresenter={openPresenterWindow}
      />

      {/* In-slide Presenter Notes Modal (for single-screen testing) */}
      <NotesModal
        isOpen={showNotes}
        onClose={() => setShowNotes(false)}
        currentSlide={currentSlide}
        totalSlides={totalSlides}
        title={currentSlideData.title}
        notes={currentSlideData.notes}
      />
    </div>
  );
}
