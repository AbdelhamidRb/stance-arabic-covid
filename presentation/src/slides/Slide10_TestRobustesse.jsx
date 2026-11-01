import React from 'react';

export default function Slide10_TestRobustesse() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 4 Pillars Grid (Sober & High Contrast) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 flex-1 min-h-0">
        {/* Pillar 1 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
              1. Décachetage Officiel (619 tweets inconnus)
            </div>
            <div className="text-xl font-extrabold text-slate-900 mt-1">F₁ = 0.7947</div>
            <div className="text-slate-500 font-mono text-[11px] mt-0.5">
              Intervalle Bootstrap 95% : [0.7626 ; 0.8273]
            </div>
            <p className="text-slate-600 mt-1 leading-snug">
              Exactitude de <strong>77.06%</strong> sur un jeu de test jamais vu par le modèle pendant toute la recherche.
            </p>
          </div>
        </div>

        {/* Pillar 2 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
              2. Test Statistique de McNemar
            </div>
            <div className="text-base font-bold text-slate-800 mt-1 font-mono">
              χ² = 0.2388 • p-valeur = 0.625
            </div>
            <p className="text-slate-600 mt-1 leading-snug">
              Comme <strong>p &gt; 0.05</strong>, l'écart d'erreurs entre MARBERT et AraBERT sur le test n'est pas statistiquement significatif : les deux échouent sur les mêmes cas extrêmes.
            </p>
          </div>
        </div>

        {/* Pillar 3 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
              3. Stress-Test d'Ablation des Mots-Clés
            </div>
            <div className="text-xl font-extrabold text-blue-900 mt-1">
              93.23% de réussite conservée
            </div>
            <p className="text-slate-600 mt-1 leading-snug">
              En masquant volontairement les mots évidents (<em>vaccin, corona, femme, numérique</em>), le score ne baisse que de <strong>6.77 points</strong>.
            </p>
          </div>
          <div className="text-[11px] text-slate-500">
            Preuve que le modèle comprend le raisonnement et non des mots isolés.
          </div>
        </div>

        {/* Pillar 4 */}
        <div className="border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs">
          <div>
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wide">
              4. Transfert vers Sujet Inconnu (Cross-Target)
            </div>
            <div className="space-y-1 mt-1 text-[11px]">
              <div className="flex justify-between">
                <span>Vers Droits des Femmes :</span>
                <span className="font-bold text-slate-800">F₁ = 0.8316 (Excellent)</span>
              </div>
              <div className="flex justify-between">
                <span>Vers Transition Numérique :</span>
                <span className="font-bold text-slate-800">F₁ = 0.7516</span>
              </div>
              <div className="flex justify-between">
                <span>Vers Vaccin Covid :</span>
                <span className="font-bold text-slate-500">F₁ = 0.6573 (Chute lexicale)</span>
              </div>
            </div>
          </div>
          <div className="text-[11px] text-slate-500">
            Le vocabulaire médical spécialisé se transfère mal sans exemples préalables.
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
            • <strong>Preuve de solidité :</strong> Si on efface les mots "tricheurs" (comme effacer le mot "vaccin"), le modèle trouve quand même la bonne réponse dans <strong>93% des cas</strong>. Il analyse donc la logique de la phrase.
          </li>
          <li>
            • <strong>Nouvelles cibles :</strong> Le modèle s'adapte très bien aux débats de société (droits, liberté), mais bloque sur les sujets médicaux pointus sans entraînement spécifique.
          </li>
        </ul>
      </div>
    </div>
  );
}
