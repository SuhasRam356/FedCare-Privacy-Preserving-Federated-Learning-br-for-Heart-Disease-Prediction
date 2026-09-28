import React, { useState } from 'react';
import { Building2, Users, HeartPulse, Activity, ShieldCheck, ArrowRight, TrendingUp } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { HOSPITALS_INFO } from '../data/repositoryData';

export const HospitalDetailView: React.FC = () => {
  const [selectedId, setSelectedId] = useState<number>(1);
  const hospital = HOSPITALS_INFO.find(h => h.id === selectedId) || HOSPITALS_INFO[0];

  return (
    <div className="space-y-6">
      {/* Node Selector Pills */}
      <div className="flex flex-wrap items-center gap-2 p-1.5 rounded-2xl bg-white/70 border border-slate-200/80 backdrop-blur-xl shadow-xs">
        {HOSPITALS_INFO.map(h => (
          <button
            key={h.id}
            onClick={() => setSelectedId(h.id)}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
              selectedId === h.id
                ? 'bg-gradient-to-r from-cyan-600 to-cyan-700 text-white shadow-sm shadow-cyan-700/25'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
            }`}
          >
            <Building2 className="size-3.5" />
            <span>H{h.id}: {h.name}</span>
          </button>
        ))}
      </div>

      {/* Main Node Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Hospital Summary */}
        <div className="space-y-6">
          <FrostedCard
            title={hospital.name}
            subtitle={`Institutional ID: Hospital ${hospital.id} · ${hospital.region}`}
            badge={hospital.status}
            badgeColor={hospital.id === 2 ? 'bg-amber-100 text-amber-800' : 'bg-cyan-100 text-cyan-800'}
          >
            <div className="space-y-4 text-xs">
              <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-2">
                <div className="flex justify-between items-center text-slate-500">
                  <span>Total Cohort Size:</span>
                  <strong className="text-slate-900 font-mono">{hospital.samples.toLocaleString()} records</strong>
                </div>
                <div className="flex justify-between items-center text-slate-500">
                  <span>Train / Test Split:</span>
                  <span className="font-mono text-slate-700">{hospital.train} train / {hospital.test} test</span>
                </div>
                <div className="flex justify-between items-center text-slate-500">
                  <span>Disease Prevalence:</span>
                  <span className={`font-mono font-bold ${hospital.id === 2 ? 'text-amber-700' : 'text-cyan-800'}`}>
                    {(hospital.prevalence * 100).toFixed(1)}% positive
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                <span className="font-bold text-slate-800 uppercase tracking-wider text-[10px]">
                  Clinical Context & Behavior:
                </span>
                <p className="text-slate-600 leading-relaxed">
                  {hospital.id === 1 && "General urban metropolitan hospital serving a broad demographic cross-section. Balanced gender and symptom mix with typical coronary artery disease risk profiles."}
                  {hospital.id === 2 && "St. Jude Heart Institute represents an extreme low-prevalence outlier (4.8% disease rate). In standard federated averaging without FedProx, its gradients risk being washed out by higher-prevalence centers."}
                  {hospital.id === 3 && "Suburban clinic with elevated average age and higher resting blood pressure, exhibiting moderate non-IID characteristics."}
                  {hospital.id === 4 && "Academic medical center specializing in complex coronary presentations. Targeted as a malicious actor in Phase 4 security attack simulations."}
                  {hospital.id === 5 && "High-volume center serving a severe pathology cohort with 46.3% disease prevalence, heavily influencing global consensus."}
                  {hospital.id === 6 && "Rural health network with high smoker incidence and diabetes prevalence, benefiting substantially from cross-institutional collaborative learning."}
                </p>
              </div>
            </div>
          </FrostedCard>
        </div>

        {/* Right 2 Columns: Metric Comparison & Features */}
        <div className="lg:col-span-2 space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
              <span className="text-xs text-slate-500">Local-Only Isolated AUC</span>
              <p className="text-2xl font-bold font-heading text-slate-700 font-mono">
                {hospital.local_auc.toFixed(4)}
              </p>
              <p className="text-[11px] text-slate-400">Trained on own {hospital.train} records</p>
            </div>

            <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
              <span className="text-xs text-slate-500">Federated Consensus AUC</span>
              <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">
                {hospital.fed_auc.toFixed(4)}
              </p>
              <p className="text-[11px] text-cyan-600 font-medium">Evaluated with global weights</p>
            </div>

            <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
              <span className="text-xs text-slate-500">Collaboration Uplift</span>
              <p className={`text-2xl font-bold font-heading font-mono ${hospital.diff.startsWith('+') ? 'text-emerald-700' : 'text-slate-600'}`}>
                {hospital.diff}
              </p>
              <p className="text-[11px] text-emerald-600 font-medium">Net performance difference</p>
            </div>
          </div>

          <FrostedCard
            title="Silo Security & Aggregation Guarantees"
            subtitle="How this hospital interacts with the central server each round"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/70 space-y-1.5">
                <span className="font-semibold text-slate-900 flex items-center gap-1.5">
                  <ShieldCheck className="size-4 text-cyan-600" />
                  What Stays at Hospital {hospital.id}
                </span>
                <ul className="list-disc list-inside space-y-1 text-slate-600 text-[11px]">
                  <li>All {hospital.samples.toLocaleString()} raw patient clinical records and vitals</li>
                  <li>Local patient identifiers, hospital databases, and CSV data</li>
                  <li>Local batch loaders and backward loss gradients</li>
                  <li>Optimizer momentum states and intermediate activations</li>
                </ul>
              </div>

              <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/70 space-y-1.5">
                <span className="font-semibold text-slate-900 flex items-center gap-1.5">
                  <Activity className="size-4 text-emerald-600" />
                  What Leaves for the Server
                </span>
                <ul className="list-disc list-inside space-y-1 text-slate-600 text-[11px]">
                  <li>Only the 3,042 weight update parameters (11.88 KB payload)</li>
                  <li>Client sample weight coefficient for FedAvg aggregation</li>
                  <li>SecAgg pairwise masking shares (simulated encryption)</li>
                  <li>Differential privacy additive noise perturbations</li>
                </ul>
              </div>
            </div>
          </FrostedCard>
        </div>
      </div>
    </div>
  );
};
