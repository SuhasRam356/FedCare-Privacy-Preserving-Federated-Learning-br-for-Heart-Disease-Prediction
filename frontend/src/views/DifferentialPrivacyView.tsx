import React from 'react';
import { EyeOff, ShieldCheck, AlertTriangle, TrendingDown } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { DP_SWEEP_DATA } from '../data/repositoryData';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from 'recharts';

export const DifferentialPrivacyView: React.FC = () => {
  const chartData = DP_SWEEP_DATA.map(d => ({
    name: d.epsilon === 'inf' ? 'No DP' : `ε=${d.epsilon}`,
    auc: Number(d.auc.toFixed(4)),
    accuracy: Number(d.accuracy.toFixed(4)),
    worst_auc: Number(d.worst_auc.toFixed(4)),
    noise: d.noise
  }));

  return (
    <div className="space-y-6">
      {/* Overview Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Unperturbed Baseline (No DP)</span>
          <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">0.8476 AUC</p>
          <p className="text-[11px] text-cyan-600">Full clinical utility, zero noise</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Weak Privacy (ε = 10.0)</span>
          <p className="text-2xl font-bold font-heading text-amber-700 font-mono">0.6112 AUC</p>
          <p className="text-[11px] text-amber-600">Initial privacy boundary degradation</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Strong Privacy (ε = 1.0)</span>
          <p className="text-2xl font-bold font-heading text-rose-700 font-mono">0.5052 AUC</p>
          <p className="text-[11px] text-rose-600">Noise overwhelms tabular gradient signal</p>
        </div>
      </div>

      {/* Trade-off Curve */}
      <FrostedCard
        title="Privacy-Utility Tradeoff Frontier"
        subtitle="Empirical evaluation of additive Gaussian noise across target privacy budgets (epsilon)"
        badge="Phase 4 DP Sweep"
      >
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 15, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} />
              <YAxis domain={[0.3, 0.9]} tick={{ fontSize: 11, fill: '#64748b' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
              <Line type="monotone" dataKey="auc" name="Global Test AUC" stroke="#0891b2" strokeWidth={3} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="accuracy" name="Test Accuracy" stroke="#059669" strokeWidth={2} strokeDasharray="3 3" />
              <Line type="monotone" dataKey="worst_auc" name="Worst Hospital AUC" stroke="#f59e0b" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </FrostedCard>

      {/* Data Table & Methodological Clarification */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <FrostedCard
            title="Empirical Differential Privacy Results"
            subtitle="Extracted from results/phase4_dp_sweep.csv"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                    <th className="py-2.5 pr-3">Privacy Target (ε)</th>
                    <th className="py-2.5 px-3">Noise Mult (σ)</th>
                    <th className="py-2.5 px-3">Privacy Regime</th>
                    <th className="py-2.5 px-3">Test Accuracy</th>
                    <th className="py-2.5 pl-3">Test AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {DP_SWEEP_DATA.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-3 pr-3 font-mono font-bold text-slate-800">
                        {row.epsilon === 'inf' ? '∞ (No Noise)' : `ε = ${row.epsilon}`}
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-600">{row.noise.toFixed(2)}</td>
                      <td className="py-3 px-3">
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                          row.regime.includes('No Privacy') ? 'bg-slate-100 text-slate-700' :
                          row.regime.includes('Weak') ? 'bg-amber-100 text-amber-800' :
                          'bg-cyan-100 text-cyan-800'
                        }`}>
                          {row.regime}
                        </span>
                      </td>
                      <td className="py-3 px-3 font-mono text-slate-700">{row.accuracy.toFixed(4)}</td>
                      <td className="py-3 pl-3 font-mono font-bold text-cyan-800">{row.auc.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </FrostedCard>
        </div>

        {/* Academic Disclosure */}
        <div>
          <FrostedCard
            title="Privacy Accountant Details"
            subtitle="Transparent research limitations"
            badge="Academic Integrity"
          >
            <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
              <p>
                In clinical tabular models with only 3,042 parameters, injecting noise calibrated for tight theoretical epsilon (<span className="font-mono text-slate-800 font-semibold">ε &lt; 1.0</span>) rapidly collapses diagnostic utility to near-random guessing (0.5052 AUC).
              </p>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1">
                <span className="font-semibold text-slate-800 block text-[11px]">Accounting Method:</span>
                <p className="text-[11px] text-slate-500">
                  Implements update-level additive Gaussian noise with naive Gaussian bounds, rather than sub-sampled patient-level DP-SGD (e.g. Opacus) or Renyi Differential Privacy (RDP).
                </p>
              </div>
            </div>
          </FrostedCard>
        </div>
      </div>
    </div>
  );
};
