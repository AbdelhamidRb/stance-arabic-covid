import React from 'react';

export default function Slide06_NLI() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* Principle Box */}
      <div className="border border-slate-200 rounded-lg p-3 bg-white">
        <div className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-1">
          Le concept NLI : Transformer la détection d'opinion en test logique
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs text-center mt-2">
          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
            <span className="text-[10px] text-slate-500 font-bold block uppercase">1. Prémisse</span>
            <span className="font-semibold text-slate-800">Le Tweet en arabe</span>
          </div>
          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
            <span className="text-[10px] text-slate-500 font-bold block uppercase">2. Hypothèse candidate</span>
            <span className="font-semibold text-slate-800">« L'auteur soutient [Cible] »</span>
          </div>
          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
            <span className="text-[10px] text-slate-500 font-bold block uppercase">3. Modèle mDeBERTa</span>
            <span className="font-semibold text-slate-800">Probabilité d'implication (Entailment)</span>
          </div>
        </div>
      </div>

      {/* 3 Languages Comparison Cards */}
      <div>
        <div className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-2">
          Comparaison selon la langue des hypothèses logiques
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Arabic */}
          <div className="border border-blue-300 rounded-lg p-3 bg-blue-50/40 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-bold text-blue-900">
                <span>🇸🇦 Hypothèse en Arabe</span>
                <span className="text-[10px] font-mono bg-blue-100 px-1.5 py-0.5 rounded">Recommandé</span>
              </div>
              <p className="font-arabic text-xs text-slate-800 my-1.5 text-right font-medium" dir="rtl">
                « الكاتب يدعم اللقاح »
              </p>
            </div>
            <div className="text-xl font-bold text-blue-900 mt-1">F₁ = 0.6925</div>
          </div>

          {/* English */}
          <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-bold text-slate-700">
                <span>🇬🇧 Hypothèse en Anglais</span>
                <span className="text-[10px] text-slate-400 font-mono">-12.3 pts</span>
              </div>
              <p className="text-xs text-slate-500 italic my-1.5">
                "The author supports the vaccine"
              </p>
            </div>
            <div className="text-xl font-bold text-slate-700 mt-1">F₁ = 0.5693</div>
          </div>

          {/* French */}
          <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-bold text-slate-700">
                <span>🇫🇷 Hypothèse en Français</span>
                <span className="text-[10px] text-slate-400 font-mono">-14.5 pts</span>
              </div>
              <p className="text-xs text-slate-500 italic my-1.5">
                « L'auteur soutient le vaccin »
              </p>
            </div>
            <div className="text-xl font-bold text-slate-700 mt-1">F₁ = 0.5471</div>
          </div>
        </div>
      </div>

      {/* 💡 En clair : Mots simples */}
      <div className="bg-slate-50 border-l-4 border-blue-600 rounded-r-lg p-3">
        <div className="text-xs font-bold text-blue-900 uppercase tracking-wide mb-1">
          💡 En clair (l'essentiel à retenir)
        </div>
        <ul className="text-xs text-slate-700 space-y-1 leading-relaxed">
          <li>
            • <strong>Zéro entraînement nécessaire :</strong> Le modèle NLI classe les tweets tout de suite, sans avoir besoin d'apprendre sur notre base (score honorable de <strong>0.69</strong>).
          </li>
          <li>
            • <strong>Le piège de la traduction :</strong> Si on formule l'hypothèse en anglais ou en français, le score s'effondre de plus de 12 points. Il faut impérativement rester dans la langue arabe pour comprendre les subtilités du tweet.
          </li>
        </ul>
      </div>
    </div>
  );
}
