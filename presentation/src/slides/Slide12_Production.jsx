import React from 'react';

const productionTable = [
  { model: 'Char-CNN (Caractères)', f1: '0.7368', size: '1.2 Mo', lat: '3.8 ms (CPU)', target: '📱 Mobile, IoT, CPU économique', isEdge: true },
  { model: 'TF-IDF + LogReg', f1: '0.7278', size: '8.5 Mo', lat: '0.4 ms (CPU)', target: 'Filtrage ultrarapide de masse' },
  { model: 'AraBERT Fine-tuné', f1: '0.7973', size: '540 Mo', lat: '38 ms (CPU)', target: 'Serveur classique' },
  { model: 'MARBERTv2 Fine-tuné', f1: '0.8225', size: '650 Mo', lat: '< 4 ms (GPU)', target: 'Serveur GPU' },
  { model: 'MARBERTv2 Multi-Tâche', f1: '0.8536', size: '650 Mo', lat: '< 4 ms (GPU)', target: '☁️ Cloud, Veille stratégique (SOTA)', isSota: true },
];

export default function Slide12_Production() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* Two Clear Choices */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {/* Choice 1: Cloud */}
        <div className="border border-blue-300 rounded-lg p-3 bg-blue-50/30 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-blue-900 uppercase">☁️ Choix Cloud / Serveurs GPU</span>
              <span className="text-[10px] font-mono bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded font-bold">Max Qualité</span>
            </div>
            <div className="text-base font-bold text-slate-900 mt-1">
              MARBERTv2 Multi-Tâche (F₁ = 0.8536)
            </div>
            <div className="text-xs text-slate-600 mt-1 leading-snug">
              • Latence GPU : <strong>&lt; 4 ms</strong> par tweet (très rapide).<br />
              • Comprend le sarcasme et l'argot grâce aux tâches auxiliaires.<br />
              • Idéal pour les ministères, grands médias et veille de crise.
            </div>
          </div>
        </div>

        {/* Choice 2: Edge */}
        <div className="border border-slate-300 rounded-lg p-3 bg-slate-50/70 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center text-xs">
              <span className="font-bold text-slate-800 uppercase">📱 Choix Embarqué / CPU Économique</span>
              <span className="text-[10px] font-mono bg-slate-200 text-slate-700 px-1.5 py-0.5 rounded font-bold">Max Frugalité</span>
            </div>
            <div className="text-base font-bold text-slate-900 mt-1">
              Char-CNN par Caractères (F₁ = 0.7368)
            </div>
            <div className="text-xs text-slate-600 mt-1 leading-snug">
              • Poids plume : <strong>1.2 Mo</strong> seulement (540× plus petit).<br />
              • Latence CPU : <strong>3.8 ms</strong> sur un simple processeur.<br />
              • 0% de mots inconnus, tourne sur mobile et coûte 0€ de GPU.
            </div>
          </div>
        </div>
      </div>

      {/* Synthesis Table */}
      <div className="border border-slate-200 rounded-lg overflow-hidden bg-white">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
            <tr>
              <th className="py-2 px-3">Modèle</th>
              <th className="py-2 px-3 text-center">F₁ Stance</th>
              <th className="py-2 px-3 text-center">Taille Disque</th>
              <th className="py-2 px-3 text-center">Vitesse</th>
              <th className="py-2 px-3">Environnement de Déploiement</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-700">
            {productionTable.map((row) => (
              <tr
                key={row.model}
                className={
                  row.isSota
                    ? 'bg-blue-50/40 font-bold text-blue-950'
                    : row.isEdge
                    ? 'bg-slate-50 font-bold text-slate-900'
                    : 'hover:bg-slate-50'
                }
              >
                <td className="py-1.5 px-3">{row.model}</td>
                <td className="py-1.5 px-3 text-center font-mono font-bold">{row.f1}</td>
                <td className="py-1.5 px-3 text-center font-mono">{row.size}</td>
                <td className="py-1.5 px-3 text-center font-mono">{row.lat}</td>
                <td className="py-1.5 px-3 text-slate-600">{row.target}</td>
              </tr>
            ))}
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
            • <strong>Deux réponses industrielles claires :</strong> Selon le budget et les machines, nous avons la solution parfaite.
          </li>
          <li>
            • <strong>Pour la précision maximale :</strong> MARBERT Multi-Tâche (0.8536) sur serveur.
          </li>
          <li>
            • <strong>Pour l'économie maximale :</strong> Le Char-CNN (1.2 Mo, 3 ms sur CPU ordinaire, 0€ de serveur).
          </li>
        </ul>
      </div>
    </div>
  );
}
