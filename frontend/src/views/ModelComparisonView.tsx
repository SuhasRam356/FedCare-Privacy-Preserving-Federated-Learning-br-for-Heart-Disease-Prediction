import React from 'react';
import { GitCompare, Award, Sparkles, CheckCircle2 } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { MODEL_BENCHMARKS } from '../data/repositoryData';
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Tooltip
} from 'recharts';

export const ModelComparisonView: React.FC = () => {
  const radarData = [
    { metric: 'Test AUC', centralized: 95, local: 78, fedavg: 94, fedper: 99 },
    { metric: 'Patient Privacy', centralized: 10, local: 100, fedavg: 90, fedper: 90 },
    { metric: 'Heterogeneity Handling', centralized: 85, local: 60, fedavg: 82, fedper: 96 },
    { metric: 'Worst-Client Equity', centralized: 88, local: 65, fedavg: 80, fedper: 95 },
    { metric: 'Network Bandwidth Efficiency', centralized: 10, local: 100, fedavg: 88, fedper: 88 },
  ];

  return (
    <div className="space-y-6">
      {/* Visual Radar Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <FrostedCard
            title="Multi-Dimensional Benchmark Evaluation"
            subtitle="Comparing federated architectures against theoretical centralized and isolated baselines"
            badge="SOTA Analysis"
          >
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#cbd5e1" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: '#475569', fontWeight: 600 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Radar name="Centralized (Upper Bound)" dataKey="centralized" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.15} />
                  <Radar name="Local-Only (Isolated)" dataKey="local" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                  <Radar name="FedAvg (Standard FL)" dataKey="fedavg" stroke="#0891b2" fill="#0891b2" fillOpacity={0.25} />
                  <Radar name="FedPer (Personalized SOTA)" dataKey="fedper" stroke="#059669" fill="#059669" fillOpacity={0.35} />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '12px',
                      border: '1px solid #cbd5e1',
                      fontSize: '12px'
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </FrostedCard>
        </div>

        {/* SOTA Summary Callout */}
        <div className="space-y-4">
          <div className="p-6 rounded-2xl bg-gradient-to-br from-cyan-600 to-cyan-800 text-white shadow-lg space-y-3">
            <span className="text-[11px] font-bold uppercase tracking-wider text-cyan-200 block">
              Federated SOTA Leader
            </span>
            <h3 className="font-heading font-extrabold text-2xl">FedPer Decoupled</h3>
            <p className="text-xs text-cyan-100 leading-relaxed">
              Achieves <strong>0.8650 AUC</strong> by keeping the final classification head strictly private and local to each hospital, allowing customization to local patient disease prevalence while leveraging shared global cardiac representations.
            </p>
            <div className="pt-2 border-t border-cyan-500/50 flex justify-between items-center text-xs font-mono">
              <span>AUC vs Centralized:</span>
              <span className="font-bold text-emerald-300">+0.0170 higher</span>
            </div>
          </div>

          <FrostedCard title="Architectural Takeaways">
            <ul className="space-y-2 text-xs text-slate-600">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span><strong>FedAvg:</strong> Recovers 94.7% of the centralized performance gap without pooling records.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span><strong>FedProx:</strong> Best choice when hospital hardware is heterogeneous or sample sizes vary.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span><strong>Tree Ensembles:</strong> Soft-voting provides transparent non-neural baseline (AUC 0.8310).</span>
              </li>
            </ul>
          </FrostedCard>
        </div>
      </div>

      {/* SOTA Benchmark Table */}
      <FrostedCard
        title="Complete Project Benchmark Matrix"
        subtitle="Comparing all model families evaluated across Phase 1 through Phase 7"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Model Architecture / Strategy</th>
                <th className="py-2.5 px-3">Test Accuracy</th>
                <th className="py-2.5 px-3">Test AUC</th>
                <th className="py-2.5 px-3">Privacy Guarantee</th>
                <th className="py-2.5 pl-3">Research Characteristics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {MODEL_BENCHMARKS.map((m, i) => (
                <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                  <td className="py-3 pr-4 font-bold text-slate-800 flex items-center gap-2">
                    {m.name.includes('FedPer') && <Sparkles className="size-3.5 text-cyan-600" />}
                    <span>{m.name}</span>
                  </td>
                  <td className="py-3 px-3 font-mono text-slate-700">{m.accuracy.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono font-bold text-cyan-800">{m.auc.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-600">{m.privacy}</td>
                  <td className="py-3 pl-3 text-slate-500 text-[11px]">{m.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
