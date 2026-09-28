import React, { useState } from 'react';
import { Skull, Shield, ShieldCheck, AlertOctagon, CheckCircle2, TrendingDown } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { ATTACK_DEFENSE_MATRIX } from '../data/repositoryData';

export const AttacksDefensesView: React.FC = () => {
  const [filter, setFilter] = useState<'all' | 'clean' | 'label_flip' | 'model_poison'>('all');

  const filteredData = filter === 'all'
    ? ATTACK_DEFENSE_MATRIX
    : ATTACK_DEFENSE_MATRIX.filter(r => r.attack_type === filter);

  return (
    <div className="space-y-6">
      {/* Overview Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Label-Flipping Impact</span>
            <AlertOctagon className="size-3.5 text-amber-600" />
          </div>
          <p className="text-2xl font-bold font-heading text-amber-700 font-mono">-0.7% AUC</p>
          <p className="text-[11px] text-slate-500">Hospitals 4 & 5 invert diagnosis labels</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Model Poisoning Impact</span>
            <Skull className="size-3.5 text-rose-600" />
          </div>
          <p className="text-2xl font-bold font-heading text-rose-700 font-mono">0.5000 AUC</p>
          <p className="text-[11px] text-rose-600">FedAvg collapses to random coin toss</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Multi-Krum Defense</span>
            <ShieldCheck className="size-3.5 text-emerald-600" />
          </div>
          <p className="text-2xl font-bold font-heading text-emerald-700 font-mono">0.8475 AUC</p>
          <p className="text-[11px] text-emerald-600">100% resilience against sign-flip poisoning</p>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex items-center gap-2 p-1.5 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-xs w-fit">
        <button
          onClick={() => setFilter('all')}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
            filter === 'all' ? 'bg-cyan-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          All Scenarios (12)
        </button>
        <button
          onClick={() => setFilter('clean')}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
            filter === 'clean' ? 'bg-cyan-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Clean Baselines
        </button>
        <button
          onClick={() => setFilter('label_flip')}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
            filter === 'label_flip' ? 'bg-cyan-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Label-Flipping Attacks
        </button>
        <button
          onClick={() => setFilter('model_poison')}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
            filter === 'model_poison' ? 'bg-cyan-600 text-white shadow-xs' : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Model Poisoning (Sign-Flip)
        </button>
      </div>

      {/* Attack Defense Matrix Table */}
      <FrostedCard
        title="Byzantine Defense Evaluation Matrix"
        subtitle="Empirical robustness benchmark from results/phase4_attack_defense_matrix.csv"
        badge="Phase 4 Security"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-3">Attack Scenario</th>
                <th className="py-2.5 px-3">Defense Aggregator</th>
                <th className="py-2.5 px-3">Test Accuracy</th>
                <th className="py-2.5 px-3">Consensus AUC</th>
                <th className="py-2.5 px-3">Worst Hosp AUC</th>
                <th className="py-2.5 pl-3">Defense Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredData.map((row, i) => {
                const isCatastrophic = row.auc === 0.5;
                const isProtected = row.auc >= 0.84;
                return (
                  <tr
                    key={i}
                    className={`hover:bg-slate-50/60 transition-colors ${
                      isCatastrophic ? 'bg-rose-50/30' : ''
                    }`}
                  >
                    <td className="py-3 pr-3 font-semibold text-slate-800">{row.attack}</td>
                    <td className="py-3 px-3 font-mono font-medium text-slate-700">{row.strategy}</td>
                    <td className="py-3 px-3 font-mono text-slate-700">{row.accuracy.toFixed(4)}</td>
                    <td className={`py-3 px-3 font-mono font-bold ${
                      isCatastrophic ? 'text-rose-700 text-sm' : 'text-cyan-800'
                    }`}>
                      {row.auc.toFixed(4)}
                    </td>
                    <td className="py-3 px-3 font-mono text-slate-600">{row.worst_auc.toFixed(4)}</td>
                    <td className="py-3 pl-3">
                      <span className={`px-2.5 py-1 rounded-full text-[11px] font-semibold inline-flex items-center gap-1.5 ${
                        isCatastrophic
                          ? 'bg-rose-100 text-rose-800'
                          : isProtected
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-amber-100 text-amber-800'
                      }`}>
                        {isCatastrophic ? <Skull className="size-3" /> : <ShieldCheck className="size-3" />}
                        {row.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
