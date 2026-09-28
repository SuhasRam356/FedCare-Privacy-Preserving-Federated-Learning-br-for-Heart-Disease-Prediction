import React, { useState } from 'react';
import { Activity, TrendingUp, Sliders, ChevronRight } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { FEDAVG_ROUNDS } from '../data/repositoryData';
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

export const TrainingTrajectoryView: React.FC = () => {
  const [metric, setMetric] = useState<'auc' | 'accuracy' | 'loss' | 'hospitals'>('auc');
  const [selectedRound, setSelectedRound] = useState<number>(11);

  const roundData = FEDAVG_ROUNDS.find(r => r.round === selectedRound) || FEDAVG_ROUNDS[10];

  return (
    <div className="space-y-6">
      {/* Metric Switcher Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-xs">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mr-1">
            Display Metric:
          </span>
          <button
            onClick={() => setMetric('auc')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              metric === 'auc' ? 'bg-cyan-600 text-white shadow-xs' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            AUC Convergence
          </button>
          <button
            onClick={() => setMetric('accuracy')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              metric === 'accuracy' ? 'bg-cyan-600 text-white shadow-xs' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Accuracy Trajectory
          </button>
          <button
            onClick={() => setMetric('loss')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              metric === 'loss' ? 'bg-cyan-600 text-white shadow-xs' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            Train vs Test Loss
          </button>
          <button
            onClick={() => setMetric('hospitals')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              metric === 'hospitals' ? 'bg-cyan-600 text-white shadow-xs' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
            }`}
          >
            All 6 Hospitals AUC
          </button>
        </div>

        <div className="text-xs font-mono text-slate-500">
          Peak Performance: <strong>AUC 0.8474</strong> (Round 11)
        </div>
      </div>

      {/* Main Chart */}
      <FrostedCard
        title="Federated Optimization Trajectory across 20 Rounds"
        subtitle="Tracking consensus metrics across multi-hospital rounds under FedAvg (E=2 local epochs, lr=0.001)"
        badge="Phase 2 Core"
      >
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={FEDAVG_ROUNDS} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis dataKey="round" tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={val => `R${val}`} />
              <YAxis
                domain={
                  metric === 'loss'
                    ? [0.38, 0.46]
                    : metric === 'hospitals'
                    ? [0.75, 0.88]
                    : [0.78, 0.86]
                }
                tick={{ fontSize: 11, fill: '#64748b' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  fontSize: '12px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />

              {metric === 'auc' && (
                <>
                  <Line type="monotone" dataKey="auc" name="Global Test AUC" stroke="#0891b2" strokeWidth={3} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="hosp_avg_auc" name="Hospital Average AUC" stroke="#059669" strokeWidth={2} strokeDasharray="4 4" />
                </>
              )}

              {metric === 'accuracy' && (
                <>
                  <Line type="monotone" dataKey="accuracy" name="Global Test Accuracy" stroke="#059669" strokeWidth={3} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="hosp_avg_accuracy" name="Hospital Avg Accuracy" stroke="#0891b2" strokeWidth={2} strokeDasharray="4 4" />
                </>
              )}

              {metric === 'loss' && (
                <>
                  <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#d97706" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="test_loss" name="Test Loss" stroke="#e11d48" strokeWidth={2.5} />
                </>
              )}

              {metric === 'hospitals' && (
                <>
                  <Line type="monotone" dataKey="hosp_1_auc" name="Hospital 1" stroke="#0891b2" strokeWidth={1.8} />
                  <Line type="monotone" dataKey="hosp_2_auc" name="Hospital 2 (Outlier)" stroke="#f59e0b" strokeWidth={2.5} />
                  <Line type="monotone" dataKey="hosp_3_auc" name="Hospital 3" stroke="#8b5cf6" strokeWidth={1.8} />
                  <Line type="monotone" dataKey="hosp_4_auc" name="Hospital 4" stroke="#06b6d4" strokeWidth={1.8} />
                  <Line type="monotone" dataKey="hosp_5_auc" name="Hospital 5" stroke="#ec4899" strokeWidth={1.8} />
                  <Line type="monotone" dataKey="hosp_6_auc" name="Hospital 6" stroke="#10b981" strokeWidth={1.8} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </FrostedCard>

      {/* Round Inspector Slider & Breakdown */}
      <FrostedCard
        title="Interactive Round Inspector"
        subtitle="Select a round to view institutional metrics at that point in time"
        badge={`Round ${selectedRound}`}
      >
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <span className="text-xs font-mono font-bold text-slate-700 w-16">
              Round {selectedRound}
            </span>
            <input
              type="range"
              min={1}
              max={20}
              value={selectedRound}
              onChange={e => setSelectedRound(Number(e.target.value))}
              className="w-full accent-cyan-600 cursor-pointer h-2 bg-slate-200 rounded-lg"
            />
            <span className="text-xs font-mono text-slate-400">R20</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
              <span className="text-[11px] text-slate-500 block">Train Loss</span>
              <span className="text-lg font-mono font-bold text-slate-800">{roundData.train_loss.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
              <span className="text-[11px] text-slate-500 block">Test Loss</span>
              <span className="text-lg font-mono font-bold text-slate-800">{roundData.test_loss.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
              <span className="text-[11px] text-slate-500 block">Global Accuracy</span>
              <span className="text-lg font-mono font-bold text-slate-800">{roundData.accuracy.toFixed(4)}</span>
            </div>
            <div className="p-3 rounded-xl bg-cyan-50/80 border border-cyan-200">
              <span className="text-[11px] text-cyan-800 block font-semibold">Global Test AUC</span>
              <span className="text-lg font-mono font-bold text-cyan-900">{roundData.auc.toFixed(4)}</span>
            </div>
          </div>
        </div>
      </FrostedCard>

      {/* Trajectory Table */}
      <FrostedCard
        title="20-Round Trajectory Data Log"
        subtitle="Complete numeric record from results/rounds_fedavg.csv"
      >
        <div className="overflow-x-auto max-h-72 overflow-y-auto">
          <table className="w-full text-left text-xs">
            <thead className="sticky top-0 bg-white border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
              <tr>
                <th className="py-2 pr-3">Round</th>
                <th className="py-2 px-3">Train Loss</th>
                <th className="py-2 px-3">Test Loss</th>
                <th className="py-2 px-3">Accuracy</th>
                <th className="py-2 px-3">AUC</th>
                <th className="py-2 px-3">H1 AUC</th>
                <th className="py-2 px-3">H2 AUC</th>
                <th className="py-2 px-3">H3 AUC</th>
                <th className="py-2 px-3">H4 AUC</th>
                <th className="py-2 px-3">H5 AUC</th>
                <th className="py-2 px-3">H6 AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {FEDAVG_ROUNDS.map(r => (
                <tr
                  key={r.round}
                  className={`hover:bg-cyan-50/30 transition-colors ${
                    r.round === selectedRound ? 'bg-cyan-50/60 font-semibold' : ''
                  }`}
                >
                  <td className="py-2 pr-3 font-mono text-cyan-800">Round {r.round}</td>
                  <td className="py-2 px-3 font-mono text-slate-600">{r.train_loss.toFixed(4)}</td>
                  <td className="py-2 px-3 font-mono text-slate-600">{r.test_loss.toFixed(4)}</td>
                  <td className="py-2 px-3 font-mono text-slate-700">{r.accuracy.toFixed(4)}</td>
                  <td className="py-2 px-3 font-mono font-bold text-slate-900">{r.auc.toFixed(4)}</td>
                  <td className="py-2 px-3 font-mono text-slate-500">{r.hosp_1_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono text-amber-700">{r.hosp_2_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono text-slate-500">{r.hosp_3_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono text-slate-500">{r.hosp_4_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono text-slate-500">{r.hosp_5_auc.toFixed(3)}</td>
                  <td className="py-2 px-3 font-mono text-slate-500">{r.hosp_6_auc.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
