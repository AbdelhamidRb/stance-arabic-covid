import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceDot } from 'recharts';

const lambdaData = [
  { lambda: 'λ = 0.0\n(Mono-tâche)', val: 0.8256 },
  { lambda: 'λ = 0.3\n(Optimum 🏆)', val: 0.8536 },
  { lambda: 'λ = 0.7\n(Conflit)', val: 0.8353 },
];

export default function Slide09_MultiTask() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 2 Blocks: Architecture + Lambda Curve */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left: 3-Head Schema */}
        <div className="md:col-span-6 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="font-bold text-slate-800 uppercase tracking-wide mb-2">
              Architecture Tri-Tête à Encodeur Partagé
            </div>

            <div className="bg-slate-900 text-white p-2 rounded text-center font-bold">
              Tronc Commun : Encodeur MARBERTv2 (12 couches)
            </div>

            <div className="text-center text-slate-400 font-bold my-1">
              ⬇️ Partage des représentations ⬇️
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="border-2 border-blue-600 bg-blue-50/40 p-2 rounded text-center">
                <span className="font-bold text-blue-900 block text-[10px] uppercase">Principale</span>
                <span className="font-bold text-slate-900">Position</span>
                <span className="text-[10px] text-slate-500 block">3 classes</span>
              </div>

              <div className="border border-slate-200 bg-slate-50 p-2 rounded text-center">
                <span className="font-semibold text-slate-500 block text-[10px] uppercase">Auxiliaire</span>
                <span className="font-bold text-slate-800">Sentiment</span>
                <span className="text-[10px] text-slate-500 block">3 classes</span>
              </div>

              <div className="border border-slate-200 bg-slate-50 p-2 rounded text-center">
                <span className="font-semibold text-slate-500 block text-[10px] uppercase">Auxiliaire</span>
                <span className="font-bold text-slate-800">Sarcasme</span>
                <span className="text-[10px] text-slate-500 block">Oui / Non</span>
              </div>
            </div>
          </div>

          <div className="bg-slate-50 p-1.5 rounded border border-slate-200 text-center font-mono text-[11px] text-slate-700">
            Perte : L_total = L_position + λ · (L_sentiment + L_sarcasme)
          </div>
        </div>

        {/* Right: Clean Curve */}
        <div className="md:col-span-6 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
          <div className="flex justify-between items-center text-xs">
            <span className="font-bold text-slate-700 uppercase tracking-wide">
              Évolution du F₁ selon le coefficient λ
            </span>
            <span className="font-bold text-blue-900 font-mono">Pic à λ = 0.3</span>
          </div>

          <div className="h-36 w-full my-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lambdaData} margin={{ top: 10, right: 25, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="#e2e8f0" />
                <XAxis dataKey="lambda" tick={{ fontSize: 10, fontWeight: 500 }} />
                <YAxis domain={[0.81, 0.87]} tick={{ fontSize: 10 }} />
                <Tooltip
                  formatter={(val) => [`${val.toFixed(4)}`, 'F₁_StanceEval']}
                  contentStyle={{ borderRadius: '6px', fontSize: '12px' }}
                />
                <Line
                  type="monotone"
                  dataKey="val"
                  stroke="#1e3a8a"
                  strokeWidth={2.5}
                  dot={{ r: 5, fill: '#1e3a8a' }}
                />
                <ReferenceDot x="λ = 0.3\n(Optimum 🏆)" y={0.8536} r={6} fill="#2563eb" stroke="#fff" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-slate-500 border-t border-slate-100 pt-1 flex justify-between">
            <span>λ = 0 : 0.8256</span>
            <span className="font-bold text-blue-900">λ = 0.3 : 0.8536 (+2.8 pts) 🏆</span>
            <span>λ = 0.7 : 0.8353</span>
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
            • <strong>L'astuce gagnante :</strong> Sur les réseaux, l'ironie trompe souvent l'IA. En forçant le modèle à prédire aussi le <em>sarcasme</em>, il apprend à repérer les faux compliments.
          </li>
          <li>
            • <strong>Le dosage parfait (λ = 0.3) :</strong> On obtient le <strong>record absolu du projet à 0.8536</strong>. Si on force trop sur les tâches annexes (λ = 0.7), le modèle perd le fil et régresse.
          </li>
        </ul>
      </div>
    </div>
  );
}
