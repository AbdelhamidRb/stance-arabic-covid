import React from 'react';

const scratchModels = [
  { id: 'C1', name: 'BiLSTM Standard', f1: '0.7177', desc: 'Réseau récurrent classique (mots)' },
  { id: 'C2', name: 'BiLSTM + Attention', f1: '0.7048', desc: 'Focalisation sur les négations (لا, ليس)' },
  { id: 'C3', name: 'Mini-Transformer (avec ordre)', f1: '0.7167', desc: '2 couches, 4 têtes d\'attention' },
  { id: 'C3-bis', name: 'Ablation : SANS ordre (no PE)', f1: '0.6981', desc: 'Chute de -1.86 pt sans l\'ordre des mots' },
  { id: 'C4', name: 'Char-CNN (par Caractères)', f1: '0.7368', desc: '🏆 1er des modèles From Scratch • 0% mots inconnus', isBest: true }
];

export default function Slide07_FromScratch() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* Table of Scratch Models */}
      <div className="border border-slate-200 rounded-lg overflow-hidden bg-white">
        <table className="w-full text-xs text-left">
          <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200">
            <tr>
              <th className="py-2 px-3">Modèle</th>
              <th className="py-2 px-3 text-center">F₁_StanceEval</th>
              <th className="py-2 px-3">Description & Rôle</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-slate-800">
            {scratchModels.map((m) => (
              <tr key={m.id} className={m.isBest ? 'bg-blue-50/50 font-bold text-blue-950' : 'hover:bg-slate-50'}>
                <td className="py-2 px-3 font-semibold">
                  <span className="font-mono text-slate-400 mr-1.5">{m.id}</span>
                  {m.name}
                </td>
                <td className="py-2 px-3 text-center font-mono font-bold">
                  {m.f1}
                </td>
                <td className="py-2 px-3 text-slate-600">
                  {m.desc}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 2 Key Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
        <div className="p-3 bg-white border border-slate-200 rounded-lg">
          <span className="font-bold text-slate-800 block mb-1">
            Le Goulet d'Étranglement OOV (Mots inconnus) :
          </span>
          <p className="text-slate-600 leading-relaxed">
            Sur Twitter, les internautes écrivent avec plein de dialectes et d'abréviations. Conséquence : <strong>20.2% des mots sont totalement inconnus</strong> pour un modèle de vocabulaire classique (1 mot sur 5 remplacé par &lt;UNK&gt;).
          </p>
        </div>

        <div className="p-3 bg-white border border-slate-200 rounded-lg">
          <span className="font-bold text-slate-800 block mb-1">
            Pourquoi le Char-CNN (C4) gagne ?
          </span>
          <p className="text-slate-600 leading-relaxed">
            Il lit <strong>lettre par lettre</strong> au lieu de mot par mot. Son taux de mots inconnus est de <strong>0.0%</strong>. Il retrouve la racine sémitique des mots arabes et résiste aux fautes de frappe.
          </p>
        </div>
      </div>

      {/* 💡 En clair : Mots simples */}
      <div className="bg-slate-50 border-l-4 border-blue-600 rounded-r-lg p-3">
        <div className="text-xs font-bold text-blue-900 uppercase tracking-wide mb-1">
          💡 En clair (l'essentiel à retenir)
        </div>
        <ul className="text-xs text-slate-700 space-y-1 leading-relaxed">
          <li>
            • <strong>L'ordre des mots compte :</strong> Si on enlève l'encodage de position au Transformer, il perd 1.86 point car il ne sait plus qui fait quoi dans la phrase.
          </li>
          <li>
            • <strong>La pépite Char-CNN :</strong> En lisant les caractères bruts, ce petit réseau de <strong>1.2 Mo</strong> bat tous les autres modèles entraînés sans pré-entraînement (<strong>0.7368</strong>).
          </li>
        </ul>
      </div>
    </div>
  );
}
