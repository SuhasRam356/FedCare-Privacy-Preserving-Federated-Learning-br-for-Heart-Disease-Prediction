import React from 'react';
import { GitFork, AlertTriangle, TrendingDown, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { NON_IID_EXPERIMENTS } from '../data/repositoryData';
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

export const NonIidView: React.FC = () => {
  const chartData = NON_IID_EXPERIMENTS.map(e => ({
    name: e.config.replace('Dirichlet ', '').replace(' - ', '\n'),
    final_auc: Number(e.auc.toFixed(4)),
    worst_auc: Number(e.worst_auc.toFixed(4)),
    equity_gap: Number(e.equity_gap.toFixed(4)),
  }));

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">IID Baseline Equity Gap</span>
          <p className="text-2xl font-bold font-heading text-emerald-700 font-mono">0.0322</p>
          <p className="text-[11px] text-emerald-600">Uniform client distribution</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Severe Skew Equity Gap (α=0.1)</span>
          <p className="text-2xl font-bold font-heading text-rose-700 font-mono">0.3595</p>
          <p className="text-[11px] text-rose-600">11x wider equity gap between clients</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Hospital-Native Skew Gap</span>
          <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">0.0766</p>
          <p className="text-[11px] text-cyan-600">Real clinical prevalence variance</p>
        </div>
      </div>

      {/* Chart: Non-IID Impact Analysis */}
      <FrostedCard
        title="Impact of Data Heterogeneity on Consensus & Equity"
        subtitle="Comparing global consensus AUC vs the worst-off hospital across Dirichlet alpha parameters"
        badge="Phase 3 Non-IID"
      >
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 15, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis domain={[0.5, 0.9]} tick={{ fontSize: 11, fill: '#64748b' }} />
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
              <Bar dataKey="worst_auc" name="Worst-Hospital AUC" fill="#f59e0b" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </FrostedCard>

      {/* Experiment Data Table */}
      <FrostedCard
        title="Dirichlet Skew Experiment Matrix"
        subtitle="Numeric findings from results/phase3_non_iid_experiments.csv"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Configuration</th>
                <th className="py-2.5 px-3">Partition Type</th>
                <th className="py-2.5 px-3">Alpha</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Global AUC</th>
                <th className="py-2.5 px-3">Worst Hosp AUC</th>
                <th className="py-2.5 px-3">Equity Gap</th>
                <th className="py-2.5 pl-3">Research Finding</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {NON_IID_EXPERIMENTS.map((row, i) => (
                <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                  <td className="py-3 pr-4 font-semibold text-slate-800">{row.config}</td>
                  <td className="py-3 px-3 font-mono text-slate-500">{row.type}</td>
                  <td className="py-3 px-3 font-mono text-slate-700">{row.alpha}</td>
                  <td className="py-3 px-3 font-mono text-slate-700">{row.accuracy.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono font-bold text-cyan-800">{row.auc.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono text-amber-700">{row.worst_auc.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono font-semibold text-rose-700">{row.equity_gap.toFixed(4)}</td>
                  <td className="py-3 pl-3 text-slate-500 text-[11px]">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
