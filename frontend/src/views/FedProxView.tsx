import React from 'react';
import { Sliders, CheckCircle2, Shield, Activity, TrendingUp } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { FEDPROX_EXPERIMENTS } from '../data/repositoryData';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const FedProxView: React.FC = () => {
  const chartData = FEDPROX_EXPERIMENTS.map(e => ({
    name: e.algorithm,
    final_auc: Number(e.auc.toFixed(4)),
    worst_auc: Number(e.worst_auc.toFixed(4)),
    equity_gap: Number(e.equity_gap.toFixed(4)),
  }));

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">FedProx Optimal Parameter</span>
          <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">μ = 0.01</p>
          <p className="text-[11px] text-cyan-600">Balances local fit with consensus</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Hospital 2 Uplift with μ=0.01</span>
          <p className="text-2xl font-bold font-heading text-emerald-700 font-mono">+0.0023 AUC</p>
          <p className="text-[11px] text-emerald-600">Prevents gradient drift on outlier client</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Excessive Penalty Risk (μ=1.0)</span>
          <p className="text-2xl font-bold font-heading text-rose-700 font-mono">-0.0130 AUC</p>
          <p className="text-[11px] text-rose-600">Over-constrains local feature discovery</p>
        </div>
      </div>

      {/* Algorithmic Comparison Chart */}
      <FrostedCard
        title="Optimization Algorithms under Clinical Heterogeneity"
        subtitle="Comparing standard FedAvg against FedProx, Adaptive Optimizers, and Personalization"
        badge="Algorithm Suite"
      >
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 15, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} interval={0} angle={-15} textAnchor="end" height={50} />
              <YAxis domain={[0.65, 0.90]} tick={{ fontSize: 11, fill: '#64748b' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Bar dataKey="final_auc" name="Global Consensus AUC" fill="#0891b2" radius={[6, 6, 0, 0]} />
              <Bar dataKey="worst_auc" name="Worst Hospital AUC" fill="#f59e0b" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </FrostedCard>

      {/* Data Table */}
      <FrostedCard
        title="Optimization Benchmark Results"
        subtitle="Complete comparison from results/phase3_fedprox_experiments.csv"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Algorithm</th>
                <th className="py-2.5 px-3">Proximal μ</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Final AUC</th>
                <th className="py-2.5 px-3">Worst Hosp AUC</th>
                <th className="py-2.5 px-3">Equity Gap</th>
                <th className="py-2.5 pl-3">H2 Post-Personalization</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {FEDPROX_EXPERIMENTS.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                  <td className="py-3 pr-4 font-semibold text-slate-800">{row.algorithm}</td>
                  <td className="py-3 px-3 font-mono text-slate-600">{row.mu}</td>
                  <td className="py-3 px-3 font-mono text-slate-700">{row.accuracy.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono font-bold text-cyan-800">{row.auc.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono text-amber-700">{row.worst_auc.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono text-slate-600">{row.equity_gap.toFixed(4)}</td>
                  <td className="py-3 pl-3 font-mono font-bold text-emerald-700">{row.h2_after.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
