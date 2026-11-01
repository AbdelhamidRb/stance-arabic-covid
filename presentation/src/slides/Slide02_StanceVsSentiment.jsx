import React from 'react';

export default function Slide02_StanceVsSentiment() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 2 Comparison Cards (Clean & Sober) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="border border-slate-200 rounded-lg p-3 bg-white">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wide">
            Test 1 : Règle naïve par modèle de sentiment
          </div>
          <div className="text-xl font-bold text-slate-800 mt-1">F₁ = 0.6614</div>
          <div className="text-xs text-slate-600 mt-1">
            Hypothèse testée : <em>« Si le tweet est positif → Pour, s'il est négatif → Contre »</em>.
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Résultat : Échec avec un tiers d'erreurs.
          </div>
        </div>

        <div className="border border-slate-200 rounded-lg p-3 bg-white">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wide">
            Test 2 : Avec un sentiment humain parfait (100% exact)
          </div>
          <div className="text-xl font-bold text-slate-800 mt-1">F₁ = 0.7481 (Plafond)</div>
          <div className="text-xs text-slate-600 mt-1">
            Même si l'ordinateur connaît parfaitement l'émotion humaine du tweet à 100%.
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            Résultat : <strong>25% d'erreurs restent incompressibles</strong> !
          </div>
        </div>
      </div>

      {/* Concrete Tweet Case Study (Clean & High Contrast) */}
      <div className="border border-slate-200 rounded-lg p-3 bg-white">
        <div className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-1">
          Exemple frappant tiré du corpus réel (Cible : Vaccin Covid)
        </div>
        <p className="font-arabic text-right text-base text-slate-900 font-semibold my-2" dir="rtl">
          « هؤلاء المتخلفين الذين يرفضون اللقاح هم سبب استمرار هذه الجائحة اللعينة! تطعموا فوراً. »
        </p>
        <p className="text-xs text-slate-500 italic mb-2">
          « Ces arriérés qui refusent le vaccin sont la cause de cette maudite pandémie ! Faites-vous vacciner immédiatement. »
        </p>
        <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-100">
          <div className="bg-slate-50 p-2 rounded border border-slate-200">
            <span className="text-slate-500 block">Sentiment de surface :</span>
            <span className="font-bold text-slate-800">Très négatif (colère, insultes)</span>
          </div>
          <div className="bg-slate-50 p-2 rounded border border-slate-200">
            <span className="text-slate-500 block">Position réelle (Stance) :</span>
            <span className="font-bold text-slate-800">POUR le vaccin (Favor)</span>
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
            • <strong>L'émotion trompeuse :</strong> Un texte agressif ou en colère n'est pas forcément "Contre" le sujet. Dans l'exemple ci-dessus, l'auteur est furieux... contre ceux qui refusent le vaccin !
          </li>
          <li>
            • <strong>La règle des 25% :</strong> 1 tweet sur 4 a un sentiment opposé à son opinion idéologique.
          </li>
          <li>
            • <strong>Conclusion :</strong> On ne peut pas faire de Stance Detection avec un simple outil d'analyse de sentiment.
          </li>
        </ul>
      </div>
    </div>
  );
}
