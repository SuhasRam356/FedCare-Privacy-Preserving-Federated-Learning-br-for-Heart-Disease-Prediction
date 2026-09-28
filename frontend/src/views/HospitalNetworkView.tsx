import React from 'react';
import { Network, Building2, Users, HeartPulse, ArrowRight, ShieldCheck, Activity } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { HOSPITALS_INFO } from '../data/repositoryData';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';

export const HospitalNetworkView: React.FC = () => {
  const chartData = HOSPITALS_INFO.map(h => ({
    name: `H${h.id}`,
    fullName: h.name,
    prevalence: Number((h.prevalence * 100).toFixed(1)),
    fed_auc: Number(h.fed_auc.toFixed(4)),
    local_auc: Number(h.local_auc.toFixed(4)),
  }));

  return (
    <div className="space-y-6">
      {/* Topology Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <FrostedCard
            title="Federated Institutional Topology (6 Nodes)"
            subtitle="Hospitals operate as independent decentralized silos; only model gradients are exchanged"
            badge="Star Topology"
          >
            {/* Visual Topology Diagram */}
            <div className="relative p-6 rounded-2xl bg-gradient-to-b from-slate-50 to-cyan-50/30 border border-slate-200/70 overflow-hidden min-h-[300px] flex items-center justify-center">
              {/* Central Server */}
              <div className="relative z-10 flex flex-col items-center justify-center size-28 rounded-3xl bg-gradient-to-br from-cyan-600 to-cyan-800 text-white shadow-xl shadow-cyan-700/25 border-4 border-white">
                <Network className="size-8 text-cyan-200" />
                <span className="font-heading font-extrabold text-xs mt-1">FedCare</span>
                <span className="text-[10px] text-cyan-200 font-mono">Aggregator</span>
              </div>

              {/* Surrounding 6 Hospital Nodes */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="size-64 rounded-full border border-dashed border-cyan-300/60 animate-spin [animation-duration:60s]" />
              </div>

              {/* Grid of Hospital Badges */}
              <div className="absolute inset-0 p-4 grid grid-cols-3 grid-rows-2 justify-between items-center pointer-events-auto">
                {HOSPITALS_INFO.map((h, i) => (
                  <div
                    key={h.id}
                    className={`flex flex-col items-center justify-center p-2 rounded-xl bg-white/90 border border-slate-200 shadow-sm max-w-[130px] mx-auto ${
                      h.id === 2 ? 'border-amber-300 ring-2 ring-amber-100' : ''
                    }`}
                  >
                    <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                      <Building2 className="size-3.5 text-cyan-600" />
                      <span>H{h.id}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 font-medium truncate w-full text-center">
                      {(h.prevalence * 100).toFixed(1)}% Prev
                    </span>
                    <span className="text-[9px] font-mono text-cyan-700 font-semibold">
                      AUC {h.fed_auc.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-between text-xs text-slate-500 gap-2">
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-emerald-500" />
                Zero cross-hospital patient leakage
              </span>
              <span className="flex items-center gap-1.5">
                <span className="size-2 rounded-full bg-amber-500" />
                Hospital 2 represents natural clinical outlier (4.8% prevalence)
              </span>
            </div>
          </FrostedCard>
        </div>

        {/* Prevalence Distribution */}
        <div>
          <FrostedCard
            title="Disease Prevalence by Center"
            subtitle="Demonstrating inherent clinical Non-IID skew"
            badge="Real Skew"
          >
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={{ stroke: '#cbd5e1' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#64748b' }} unit="%" axisLine={{ stroke: '#cbd5e1' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      borderRadius: '12px',
                      border: '1px solid #cbd5e1',
                      fontSize: '12px'
                    }}
                    formatter={(val: any) => [`${val}%`, 'Prevalence']}
                  />
                  <Bar dataKey="prevalence" radius={[6, 6, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.name === 'H2' ? '#f59e0b' : '#0891b2'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 mt-2 text-center">
              Prevalence ranges from <strong>4.8%</strong> (St. Jude) up to <strong>46.3%</strong> (Beacon Health).
            </p>
          </FrostedCard>
        </div>
      </div>

      {/* Detailed Node Table */}
      <FrostedCard
        title="Node Distribution & Performance Roster"
        subtitle="Comparing standalone isolated local training vs joint federated consensus"
        badge="6 Centers"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Node ID</th>
                <th className="py-2.5 px-3">Center Name</th>
                <th className="py-2.5 px-3">Role / Region</th>
                <th className="py-2.5 px-3">Cohort Size</th>
                <th className="py-2.5 px-3">Prevalence</th>
                <th className="py-2.5 px-3">Local AUC</th>
                <th className="py-2.5 px-3">Federated AUC</th>
                <th className="py-2.5 pl-3">Net Gain</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {HOSPITALS_INFO.map(h => (
                <tr key={h.id} className="hover:bg-slate-50/60 transition-colors">
                  <td className="py-3 pr-4 font-mono font-bold text-slate-900">Hospital {h.id}</td>
                  <td className="py-3 px-3 font-semibold text-slate-800">{h.name}</td>
                  <td className="py-3 px-3 text-slate-500">{h.region}</td>
                  <td className="py-3 px-3 font-mono text-slate-600">{h.samples.toLocaleString()} pts</td>
                  <td className="py-3 px-3 font-mono">
                    <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                      h.id === 2 ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'
                    }`}>
                      {(h.prevalence * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-3 px-3 font-mono text-slate-600">{h.local_auc.toFixed(4)}</td>
                  <td className="py-3 px-3 font-mono font-bold text-cyan-800">{h.fed_auc.toFixed(4)}</td>
                  <td className="py-3 pl-3 font-mono font-bold">
                    <span className={h.diff.startsWith('+') ? 'text-emerald-600' : 'text-slate-500'}>
                      {h.diff}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
