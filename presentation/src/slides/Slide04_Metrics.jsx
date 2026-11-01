import React from 'react';

export default function Slide04_Metrics() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 3 Metric Cards (Clean & Minimalist) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Metric 1 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wide">
            1. Accuracy (Trompeuse)
          </div>
          <div className="font-mono text-xs font-semibold text-slate-800 my-1 bg-slate-50 p-1.5 rounded text-center border border-slate-200">
            N_corrects / N_total
          </div>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed">
            Très risquée sur des données déséquilibrées : un modèle paresseux peut afficher 61% sans rien comprendre.
          </p>
        </div>

        {/* Metric 2 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white">
          <div className="text-xs font-bold text-slate-500 uppercase tracking-wide">
            2. Macro-F₁ (3 Classes)
          </div>
          <div className="font-mono text-xs font-semibold text-slate-800 my-1 bg-slate-50 p-1.5 rounded text-center border border-slate-200">
            (F₁_Pour + F₁_Contre + F₁_Neutre) / 3
          </div>
          <p className="text-xs text-slate-600 mt-1 leading-relaxed">
            Pèse à égalité les 3 classes, forçant à bien détecter les 10% de tweets neutres.
          </p>
        </div>

        {/* Metric 3 */}
        <div className="border border-slate-300 rounded-lg p-3 bg-slate-50/70">
          <div className="text-xs font-bold text-blue-900 uppercase tracking-wide">
            3. F₁_StanceEval (Officielle) ⭐
          </div>
          <div className="font-mono text-xs font-bold text-blue-900 my-1 bg-white p-1.5 rounded text-center border border-blue-200">
            (F₁_Pour + F₁_Contre) / 2
          </div>
          <p className="text-xs text-slate-700 mt-1 leading-relaxed">
            Métrique de la compétition : mesure la capacité à départager les deux camps opposés.
          </p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="border border-slate-200 rounded-lg overflow-hidden bg-white">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
            <tr>
              <th className="py-2 px-3">Modèle</th>
              <th className="py-2 px-3 text-center">Accuracy (Trompeuse)</th>
              <th className="py-2 px-3 text-center">F₁ Pour</th>
              <th className="py-2 px-3 text-center">F₁ Contre</th>
              <th className="py-2 px-3 text-center font-bold text-blue-900">F₁_StanceEval (Vrai juge)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-800">
            <tr>
              <td className="py-2 px-3 font-semibold">Modèle Naïf ("Toujours Pour")</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-slate-500">61.25% ⚠️</td>
              <td className="py-2 px-3 text-center font-mono">0.7607</td>
              <td className="py-2 px-3 text-center font-mono text-slate-400">0.0000</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-slate-600 bg-slate-50">0.3804</td>
            </tr>
            <tr>
              <td className="py-2 px-3 font-semibold">TF-IDF Linéaire</td>
              <td className="py-2 px-3 text-center font-mono">75.78%</td>
              <td className="py-2 px-3 text-center font-mono">0.8242</td>
              <td className="py-2 px-3 text-center font-mono">0.6314</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-slate-800 bg-slate-50">0.7278</td>
            </tr>
            <tr className="bg-blue-50/40">
              <td className="py-2 px-3 font-bold text-blue-950">MARBERTv2 Multi-Tâche (SOTA)</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-blue-900">83.47%</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-blue-900">0.8993</td>
              <td className="py-2 px-3 text-center font-mono font-bold text-blue-900">0.8079</td>
              <td className="py-2 px-3 text-center font-mono font-extrabold text-blue-900 bg-blue-100/60">0.8536 🏆</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* 💡 En clair : Mots simples */}
      <div className="bg-slate-50 border-l-4 border-blue-600 rounded-r-lg p-3">
        <div className="text-xs font-bold text-blue-900 uppercase tracking-wide mb-1">
          💡 En clair (l'essentiel à retenir)
        </div>
        <ul className="text-xs text-slate-700 space-y-1 leading-relaxed">
          <li>
            • <strong>Pourquoi l'Accuracy est bannie :</strong> Si on répond toujours "Pour", on a 61% de réussite sans savoir lire. C'est une illusion d'efficacité.
          </li>
          <li>
            • <strong>Le vrai juge (F₁_StanceEval) :</strong> Il met 0/20 si le modèle ne trouve jamais les "Contre". Notre baseline naïve tombe ainsi à <strong>0.38</strong>, ce qui est honnête.
          </li>
        </ul>
      </div>
    </div>
  );
}
