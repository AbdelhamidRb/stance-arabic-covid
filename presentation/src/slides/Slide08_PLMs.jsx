import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

const plmData = [
  { name: 'D1 : AraBERT Gelé (CLS)', score: 0.6956, color: '#94a3b8' },
  { name: 'D1 : AraBERT Gelé (Mean Pool)', score: 0.7138, color: '#64748b' },
  { name: 'D2 : AraBERT + BiLSTM', score: 0.7748, color: '#475569' },
  { name: 'E1 : AraBERT Fine-tuné (Twitter)', score: 0.7973, color: '#334155' },
  { name: 'E1 : MARBERTv2 Fine-tuné', score: 0.8225, color: '#1e3a8a' },
];

export default function Slide08_PLMs() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* Chart & Duel */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left: Chart */}
        <div className="md:col-span-7 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
          <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Progression des Transformers Arabes (F₁_StanceEval)
          </div>

          <div className="h-40 w-full my-1">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={plmData} layout="vertical" margin={{ top: 5, right: 35, left: 10, bottom: 5 }}>
                <XAxis type="number" domain={[0.65, 0.85]} tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={180} tick={{ fontSize: 11, fontWeight: 500 }} />
                <Tooltip
                  formatter={(val) => [`${val.toFixed(4)}`, 'F₁_StanceEval']}
                  contentStyle={{ borderRadius: '6px', fontSize: '12px' }}
                />
                <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                  {plmData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-slate-500 border-t border-slate-100 pt-1 flex justify-between">
            <span>Mean Pooling &gt; CLS (+1.8 pt)</span>
            <span>Évaluation sur 3 graines (seeds 42, 123, 999)</span>
          </div>
        </div>

        {/* Right: AraBERT vs MARBERTv2 */}
        <div className="md:col-span-5 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="font-bold text-slate-800 uppercase tracking-wide mb-2">
              Le Duel des Deux Grands Transformers
            </div>

            <div className="p-2 bg-slate-50 border border-slate-200 rounded mb-2">
              <span className="font-bold text-slate-800 block">AraBERTv02-Twitter (0.7973)</span>
              <p className="text-slate-600 mt-0.5 leading-snug">
                Entraîné sur des articles de presse officiels et des tweets institutionnels.
              </p>
            </div>

            <div className="p-2 bg-blue-50/50 border border-blue-200 rounded">
              <span className="font-bold text-blue-950 block">MARBERTv2 (0.8225 🏆)</span>
              <p className="text-blue-900 mt-0.5 leading-snug">
                Entraîné sur <strong>1 Milliard de vrais tweets</strong> de tous les pays arabes (Golfe, Levant, Maghreb).
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
            • <strong>L'arabe formel vs l'arabe de la rue :</strong> AraBERT connaît l'arabe des journaux, alors que MARBERTv2 a lu 1 milliard de tweets informels. C'est pour cela que MARBERT gagne sur 100% des tests.
          </li>
          <li>
            • <strong>Le fine-tuning est indispensable :</strong> Entraîner le Transformer de bout en bout fait passer le score de <strong>0.71</strong> (gelé) à <strong>0.8225</strong>.
          </li>
        </ul>
      </div>
    </div>
  );
}
