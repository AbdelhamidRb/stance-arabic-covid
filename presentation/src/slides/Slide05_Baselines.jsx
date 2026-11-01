import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const baselineData = [
  { name: 'Modèle Trivial (Majorité)', score: 0.3804, color: '#94a3b8' },
  { name: 'Règles Hashtags (72 motifs)', score: 0.4071, color: '#64748b' },
  { name: 'TF-IDF (Mots seuls)', score: 0.6885, color: '#475569' },
  { name: 'TF-IDF (Mots + Caractères)', score: 0.7278, color: '#1e3a8a' },
];

export default function Slide05_Baselines() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* Chart + Failure Case */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left: Sober Chart */}
        <div className="md:col-span-7 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
          <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Performances des Baselines (F₁_StanceEval)
          </div>

          <div className="h-40 w-full my-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={baselineData} layout="vertical" margin={{ top: 5, right: 35, left: 10, bottom: 5 }}>
                <XAxis type="number" domain={[0, 0.8]} tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={175} tick={{ fontSize: 11, fontWeight: 500 }} />
                <Tooltip
                  formatter={(val) => [`${val.toFixed(4)}`, 'F₁_StanceEval']}
                  contentStyle={{ borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                  {baselineData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-slate-500 border-t border-slate-100 pt-1 flex justify-between">
            <span>TF-IDF atteint <strong>0.7278</strong></span>
            <span>Ablation caractères : -4 points</span>
          </div>
        </div>

        {/* Right: Concrete Failure Case */}
        <div className="md:col-span-5 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="font-bold text-slate-800 uppercase tracking-wide mb-1">
              Le Piège des Corrélations Fallacieuses
            </div>
            <p className="text-slate-600 leading-snug mb-2">
              Le TF-IDF associe mécaniquement le mot <span className="font-arabic font-bold">« مؤامرة »</span> (complot) à <strong>Contre</strong>.
            </p>

            <div className="p-2 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-500 uppercase font-bold block">Tweet réel testé :</span>
              <p className="font-arabic text-sm text-slate-900 font-semibold my-1 text-right" dir="rtl">
                « نحن واثقون في العلم ولا نصدق نظرية المؤامرة الكاذبة »
              </p>
              <p className="text-[11px] text-slate-500 italic">
                « Nous croyons en la science et ne croyons PAS au complot »
              </p>
              <div className="flex justify-between items-center mt-2 pt-1 border-t border-slate-200 text-[11px]">
                <span className="text-slate-500">Prédit : <strong>CONTRE</strong> ❌</span>
                <span className="text-slate-900 font-bold">Réalité : POUR ✅</span>
              </div>
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
            • <strong>Un modèle statistique surprenant :</strong> Le simple TF-IDF (sac de mots) monte déjà à <strong>0.7278</strong> en mémorisant les mots fréquents.
          </li>
          <li>
            • <strong>Sa grosse limite :</strong> Il ne comprend pas la grammaire ni la négation (« ne pas », « faux »). Il voit le mot "complot" et classe le tweet en "Contre" même si l'auteur dit le contraire.
          </li>
        </ul>
      </div>
    </div>
  );
}
