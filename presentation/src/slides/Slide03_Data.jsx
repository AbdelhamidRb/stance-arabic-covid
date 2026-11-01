import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const classData = [
  { name: 'Pour (Favor)', count: 2145, pct: 61.25, color: '#1e3a8a' },
  { name: 'Contre (Against)', count: 1018, pct: 29.06, color: '#334155' },
  { name: 'Neutre (None)', count: 339, pct: 9.69, color: '#64748b' }
];

export default function Slide03_Data() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 3 Stats (Clean) */}
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="border border-slate-200 rounded-lg p-2.5 bg-slate-50/50">
          <div className="text-[11px] text-slate-500 font-medium">Train Interne (90%)</div>
          <div className="text-xl font-bold text-slate-800">3 151 tweets</div>
          <div className="text-[10px] text-slate-400">Pour entraîner les modèles</div>
        </div>

        <div className="border border-slate-200 rounded-lg p-2.5 bg-slate-50/50">
          <div className="text-[11px] text-slate-500 font-medium">Validation Interne (10%)</div>
          <div className="text-xl font-bold text-slate-800">351 tweets</div>
          <div className="text-[10px] text-slate-400">Pour régler les seuils</div>
        </div>

        <div className="border border-slate-300 rounded-lg p-2.5 bg-white">
          <div className="text-[11px] text-slate-700 font-bold">Test Scellé (Officiel) 🔒</div>
          <div className="text-xl font-extrabold text-blue-900">619 tweets</div>
          <div className="text-[10px] text-slate-500">Décacheté à la fin absolue</div>
        </div>
      </div>

      {/* Main Row: Chart + Pandas Bug */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left: Clean Chart */}
        <div className="md:col-span-7 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
          <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Distribution des 3 classes (Corpus Mawqif, N = 3 502)
          </div>

          <div className="h-36 w-full my-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                <XAxis type="number" domain={[0, 70]} unit="%" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11, fontWeight: 500 }} />
                <Tooltip
                  formatter={(val, name, item) => [`${item.payload.count} tweets (${item.payload.pct}%)`, 'Effectif']}
                  contentStyle={{ borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-slate-500 flex justify-between border-t border-slate-100 pt-1">
            <span>61.2% Pour</span>
            <span>29.1% Contre</span>
            <span>9.7% Neutre</span>
          </div>
        </div>

        {/* Right: Technical integrity card */}
        <div className="md:col-span-5 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="font-bold text-slate-800 uppercase tracking-wide mb-1">
              Rigueur Méthodologique
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded mb-2">
              <span className="font-bold text-slate-800 block">Bug Pandas corrigé :</span>
              <p className="text-slate-600 mt-0.5 leading-snug">
                Par défaut, la fonction de lecture supprimait les 339 tweets avec le label "None" en les prenant pour des cases vides (<code className="font-mono">NaN</code>) ! Corrigé immédiatement.
              </p>
            </div>
            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <span className="font-bold text-slate-800 block">Stratification propre :</span>
              <p className="text-slate-600 mt-0.5 leading-snug">
                Découpage en 9 strates (Cible × Position) pour que chaque split contienne exactement la même proportion de sujets.
              </p>
            </div>
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
            • <strong>Déséquilibre fort :</strong> La majorité des internautes sont "Pour" (61%). Si un modèle répond toujours "Pour", il a vite l'air bon alors qu'il triche.
          </li>
          <li>
            • <strong>Zéro triche :</strong> Les 619 tweets du test officiel ont été scellés et cachés jusqu'au tout dernier jour de l'évaluation finale.
          </li>
        </ul>
      </div>
    </div>
  );
}
