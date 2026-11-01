import React from 'react';

export default function Slide01_Intro() {
  return (
    <div className="h-full flex flex-col justify-between py-1 space-y-3">
      {/* 3 Classes Cards (Clean, Sober) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Favor */}
        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              Pour (Favor)
            </span>
            <span className="text-xs font-semibold text-slate-500">61% des tweets</span>
          </div>
          <p className="font-arabic text-right text-sm text-slate-800 font-semibold my-2" dir="rtl">
            « أنا أؤيد التطعيم بكل قوة لحماية أسرنا »
          </p>
          <div className="text-[11px] text-slate-500">
            L'auteur défend la cible (ex: le vaccin).
          </div>
        </div>

        {/* Against */}
        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              Contre (Against)
            </span>
            <span className="text-xs font-semibold text-slate-500">29% des tweets</span>
          </div>
          <p className="font-arabic text-right text-sm text-slate-800 font-semibold my-2" dir="rtl">
            « اللقاح مؤامرة تجارية ولا فائدة منه أبداً »
          </p>
          <div className="text-[11px] text-slate-500">
            L'auteur critique ou rejette la cible.
          </div>
        </div>

        {/* None */}
        <div className="border border-slate-200 rounded-lg p-3 bg-slate-50/50 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">
              Neutre (None)
            </span>
            <span className="text-xs font-semibold text-slate-500">10% des tweets</span>
          </div>
          <p className="font-arabic text-right text-sm text-slate-700 my-2" dir="rtl">
            « أعلنت وزارة الصحة اليوم عن أوقات العمل »
          </p>
          <div className="text-[11px] text-slate-500">
            Tweet purement informatif ou sans avis.
          </div>
        </div>
      </div>

      {/* Model Evolution Line (Minimalist) */}
      <div className="border border-slate-200 rounded-lg p-3 bg-white">
        <div className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-2">
          Progression des performances du projet (F₁_StanceEval)
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 text-center text-xs">
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500">Modèle Aléatoire</div>
            <div className="text-base font-bold text-slate-700">0.3804</div>
          </div>
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500">Modèle TF-IDF</div>
            <div className="text-base font-bold text-slate-800">0.7278</div>
          </div>
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500">Char-CNN (Caractères)</div>
            <div className="text-base font-bold text-slate-800">0.7368</div>
          </div>
          <div className="p-2 rounded bg-slate-50 border border-slate-200">
            <div className="text-[10px] text-slate-500">MARBERTv2 Fine-Tuné</div>
            <div className="text-base font-bold text-slate-800">0.8225</div>
          </div>
          <div className="p-2 rounded bg-blue-50 border border-blue-300">
            <div className="text-[10px] text-blue-900 font-bold">Multi-Tâche (Record)</div>
            <div className="text-base font-extrabold text-blue-900">0.8536 🏆</div>
          </div>
        </div>
      </div>

      {/* 💡 En clair : Mots simples et pas longs */}
      <div className="bg-slate-50 border-l-4 border-blue-600 rounded-r-lg p-3">
        <div className="text-xs font-bold text-blue-900 uppercase tracking-wide mb-1">
          💡 En clair (l'essentiel à retenir)
        </div>
        <ul className="text-xs text-slate-700 space-y-1 leading-relaxed">
          <li>
            • <strong>L'objectif :</strong> Déterminer automatiquement si un auteur de tweet en arabe est <strong>Pour</strong>, <strong>Contre</strong> ou <strong>Neutre</strong> vis-à-vis d'un sujet précis (Vaccin Covid, Transition Numérique, Droits des femmes).
          </li>
          <li>
            • <strong>Le défi :</strong> Sur Twitter, l'arabe mélange plusieurs dialectes, de l'argot et des fautes d'orthographe.
          </li>
          <li>
            • <strong>Notre résultat :</strong> Nous sommes passés d'un modèle de base à <strong>0.38</strong> jusqu'à un record à <strong>0.8536</strong> grâce à un modèle entraîné conjointement sur le sarcasme.
          </li>
        </ul>
      </div>
    </div>
  );
}
