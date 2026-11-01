import React from 'react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts';

const errorTaxonomy = [
  { name: 'Position Implicite / Sous-entendus', value: 33.3, color: '#1e3a8a' },
  { name: 'Sarcasme & Ironie Dialectale', value: 26.7, color: '#334155' },
  { name: 'Sentiment Opposé à l\'Opinion', value: 20.0, color: '#475569' },
  { name: 'Cible Confuse / Sujets Multiples', value: 13.3, color: '#64748b' },
  { name: 'Bruit d\'Annotation Humaine', value: 6.7, color: '#94a3b8' },
];

export default function Slide11_Errors() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 2 Blocks: Pie Chart + Concrete Tweet Examples */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-3 flex-1 min-h-0">
        {/* Left: Sober Pie Chart */}
        <div className="md:col-span-5 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between">
          <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">
            Causes des 142 erreurs sur le Test (23%)
          </div>

          <div className="h-36 w-full my-1">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={errorTaxonomy}
                  cx="50%"
                  cy="50%"
                  innerRadius={32}
                  outerRadius={58}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {errorTaxonomy.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(val, name) => [`${val}%`, name]}
                  contentStyle={{ borderRadius: '6px', fontSize: '11px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-1 text-[11px] text-slate-600 border-t border-slate-100 pt-1">
            <div className="flex justify-between"><span>• Sous-entendus / Implicite</span><span className="font-bold">33%</span></div>
            <div className="flex justify-between"><span>• Sarcasme / Ironie</span><span className="font-bold">27%</span></div>
            <div className="flex justify-between"><span>• Sentiment en désaccord</span><span className="font-bold">20%</span></div>
          </div>
        </div>

        {/* Right: 2 Concrete Tweet Failures */}
        <div className="md:col-span-7 border border-slate-200 rounded-lg p-3 bg-white flex flex-col justify-between text-xs space-y-2">
          {/* Example 1: Implicite */}
          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
              <span>Cas 1 : Position Implicite (33%)</span>
              <span>Cible : Digital</span>
            </div>
            <p className="font-arabic text-sm text-slate-900 font-medium text-right my-1 leading-snug" dir="rtl">
              « نقل الاموال مكلفة وخطرة والكاش يضيع وينسرق.. التحول للدفع الالكتروني »
            </p>
            <p className="text-[11px] text-slate-500 italic">
              L'auteur liste les dangers du liquide pour promouvoir le digital, sans compliments directs. Le modèle voit "danger, vol" et classe en <strong>Contre</strong> par erreur.
            </p>
          </div>

          {/* Example 2: Sarcasme */}
          <div className="p-2 bg-slate-50 border border-slate-200 rounded">
            <div className="flex justify-between text-[10px] font-bold text-slate-500 uppercase">
              <span>Cas 2 : Sarcasme Dialectal (27%)</span>
              <span>Cible : Vaccin</span>
            </div>
            <p className="font-arabic text-sm text-slate-900 font-medium text-right my-1 leading-snug" dir="rtl">
              « والله ما قصرتوا، كل يوم تطعيمة رابعة وخامسة! استمروا ما شاء الله »
            </p>
            <p className="text-[11px] text-slate-500 italic">
              Formules élogieuses (« Bravo, continuez ») utilisées pour se moquer des rappels vaccinaux. Le modèle prend l'éloge au 1er degré et prédit <strong>Pour</strong> au lieu de Contre.
            </p>
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
            • <strong>D'où viennent les erreurs ?</strong> Dans 60% des cas d'échec, le problème vient de l'ironie (dire le contraire de ce qu'on pense) ou des sous-entendus (argumenter sans dire explicitement "J'aime" ou "Je déteste").
          </li>
          <li>
            • <strong>C'est la frontière actuelle du TALN :</strong> Même les meilleurs LLMs ont encore besoin de contexte ou de données multimodales pour saisir le 2nd degré de certains dialectes.
          </li>
        </ul>
      </div>
    </div>
  );
}
