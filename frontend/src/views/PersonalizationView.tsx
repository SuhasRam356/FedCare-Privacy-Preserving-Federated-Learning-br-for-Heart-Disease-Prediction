import React from 'react';
import { Sparkles, CheckCircle2, Layers, Award, TrendingUp, Shield } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';

export const PersonalizationView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* SOTA Showcase Banner */}
      <div className="p-6 rounded-3xl bg-gradient-to-br from-cyan-600 via-cyan-700 to-slate-800 text-white shadow-xl shadow-cyan-900/10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/20 backdrop-blur-md text-xs font-semibold text-cyan-100">
              <Sparkles className="size-3.5" />
              Highest In-Project Performance Achieved
            </div>
            <h2 className="font-heading font-extrabold text-2xl sm:text-3xl tracking-tight text-white">
              Personalized Federated Learning (FedPer & FedBN)
            </h2>
            <p className="text-sm text-cyan-100/90 leading-relaxed">
              By decoupling the global shared representation from private local classification heads, FedPer resolves the fundamental tension between cross-institutional generalizability and local clinical customization, pushing test AUC to a state-of-the-art <strong className="text-white font-bold">0.8650</strong>.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-4">
            <div className="p-4 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 text-center min-w-[140px]">
              <span className="text-[11px] uppercase tracking-wider text-cyan-200 block font-semibold">Peak FedPer AUC</span>
              <span className="text-3xl font-extrabold font-mono text-white">0.8650</span>
              <span className="text-[10px] text-emerald-300 block font-medium">+0.017 vs FedAvg</span>
            </div>
            <div className="p-4 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 text-center min-w-[140px]">
              <span className="text-[11px] uppercase tracking-wider text-cyan-200 block font-semibold">Equity Gap</span>
              <span className="text-3xl font-extrabold font-mono text-white">0.0420</span>
              <span className="text-[10px] text-cyan-200 block font-medium">44% reduction</span>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Decoupling Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FrostedCard
          title="FedPer: Base / Head Decoupling"
          subtitle="Shared feature representation with hospital-specific final classification heads"
          badge="Architecture 1"
        >
          <div className="space-y-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3 font-mono">
              <div className="p-2.5 rounded-lg bg-cyan-50 border border-cyan-200 text-cyan-900">
                <span className="font-bold block text-[11px] text-cyan-800">SHARED GLOBALLY ACROSS FEDERATION:</span>
                Input (13) → Linear(64) + ReLU → Linear(32) + ReLU
                <span className="text-[10px] text-slate-500 block mt-0.5">Learns universally transferable cardiac biomarker representations</span>
              </div>
              <div className="flex justify-center text-slate-400 font-bold">↓ Decoupling Boundary ↓</div>
              <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900">
                <span className="font-bold block text-[11px] text-emerald-800">KEPT STRICTLY PRIVATE AT EACH HOSPITAL:</span>
                Linear(32 → 2) Classification Head (Frozen from aggregation)
                <span className="text-[10px] text-slate-500 block mt-0.5">Calibrates decision boundary to local disease prevalence</span>
              </div>
            </div>

            <ul className="space-y-2 text-slate-600">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>Prevents the severe performance drops experienced by skewed hospitals (like Hospital 2 with 4.8% prevalence).</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>Achieves the highest test accuracy (0.8250) and test AUC (0.8650) in the entire FedCare project.</span>
              </li>
            </ul>
          </div>
        </FrostedCard>

        <FrostedCard
          title="FedBN: Local Batch Normalization"
          subtitle="Hospital-specific normalization statistics mitigating feature shift"
          badge="Architecture 2"
        >
          <div className="space-y-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80 space-y-3 font-mono">
              <div className="p-2.5 rounded-lg bg-cyan-50 border border-cyan-200 text-cyan-900">
                <span className="font-bold block text-[11px] text-cyan-800">SHARED GLOBALLY:</span>
                Linear weight matrices and biases
              </div>
              <div className="flex justify-center text-slate-400 font-bold">≠ Excluded from Averaging ≠</div>
              <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900">
                <span className="font-bold block text-[11px] text-amber-800">KEPT LOCAL:</span>
                BatchNorm running mean & variance (μ_local, σ_local)
                <span className="text-[10px] text-slate-500 block mt-0.5">Accounts for equipment calibration differences between hospitals</span>
              </div>
            </div>

            <ul className="space-y-2 text-slate-600">
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>AUC reaches 0.8570, outperforming standard FedAvg without requiring any local classifier fine-tuning.</span>
              </li>
              <li className="flex items-start gap-2">
                <CheckCircle2 className="size-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>Particularly effective when clinical sensor calibrations differ across participating medical centers.</span>
              </li>
            </ul>
          </div>
        </FrostedCard>
      </div>

      {/* Comparison Table */}
      <FrostedCard
        title="Personalization vs Standard Baselines"
        subtitle="Verifying how personalization solves clinical equity"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Algorithm</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Global AUC</th>
                <th className="py-2.5 px-3">Worst Client AUC</th>
                <th className="py-2.5 px-3">Equity Gap</th>
                <th className="py-2.5 pl-3">Architecture Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              <tr className="hover:bg-slate-50/60">
                <td className="py-3 pr-4 font-semibold text-slate-800">FedAvg Baseline</td>
                <td className="py-3 px-3 font-mono text-slate-600">0.8117</td>
                <td className="py-3 px-3 font-mono text-slate-700">0.8493</td>
                <td className="py-3 px-3 font-mono text-amber-700">0.7810</td>
                <td className="py-3 px-3 font-mono text-slate-600">0.0748</td>
                <td className="py-3 pl-3 text-slate-500">Global consensus, no personalization</td>
              </tr>
              <tr className="hover:bg-slate-50/60">
                <td className="py-3 pr-4 font-semibold text-slate-800">FedBN</td>
                <td className="py-3 px-3 font-mono text-slate-600">0.8180</td>
                <td className="py-3 px-3 font-mono text-cyan-800 font-semibold">0.8570</td>
                <td className="py-3 px-3 font-mono text-slate-700">0.7980</td>
                <td className="py-3 px-3 font-mono text-emerald-700 font-semibold">0.0510</td>
                <td className="py-3 pl-3 text-slate-500">Local Batch Normalization layers</td>
              </tr>
              <tr className="hover:bg-cyan-50/40 bg-cyan-50/20">
                <td className="py-3 pr-4 font-bold text-cyan-900 flex items-center gap-1.5">
                  <Sparkles className="size-3.5 text-cyan-600" />
                  FedPer (Base/Head Split)
                </td>
                <td className="py-3 px-3 font-mono text-cyan-900 font-bold">0.8250</td>
                <td className="py-3 px-3 font-mono text-cyan-900 font-extrabold text-sm">0.8650</td>
                <td className="py-3 px-3 font-mono text-emerald-800 font-bold">0.8110</td>
                <td className="py-3 px-3 font-mono text-emerald-700 font-bold">0.0420</td>
                <td className="py-3 pl-3 text-cyan-800 font-medium">Decoupled classification head (Project SOTA)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
