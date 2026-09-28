import React from 'react';
import { Wifi, ArrowDownUp, Layers, HardDrive, CheckCircle2, Shield } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { COMMUNICATION_SPECS } from '../data/repositoryData';

export const CommunicationCostView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Model Parameters</span>
          <p className="text-2xl font-bold font-heading text-slate-900 font-mono">3,042</p>
          <p className="text-[11px] text-slate-500">13 → 64 → 32 → 2 MLP</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Per-Hospital Message</span>
          <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">11.88 KB</p>
          <p className="text-[11px] text-cyan-600">Float32 gradient vector</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Per-Round Federation Traffic</span>
          <p className="text-2xl font-bold font-heading text-slate-900 font-mono">142.59 KB</p>
          <p className="text-[11px] text-slate-500">6 Uplinks + 6 Downlinks</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Total 20-Round Traffic</span>
          <p className="text-2xl font-bold font-heading text-emerald-700 font-mono">1.671 MB</p>
          <p className="text-[11px] text-emerald-600">Entire training cycle</p>
        </div>
      </div>

      {/* Traffic Composition Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FrostedCard
          title="Network Payload Architecture"
          subtitle="Detailed byte allocation per client message transmission"
          badge="Network Efficiency"
        >
          <div className="space-y-4 text-xs">
            <div className="space-y-2">
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="font-semibold text-slate-800">Layer 1: Linear(13, 64) Weights + Biases</span>
                <span className="font-mono text-cyan-800 font-bold">3,584 bytes</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="font-semibold text-slate-800">Layer 2: Linear(64, 32) Weights + Biases</span>
                <span className="font-mono text-cyan-800 font-bold">8,320 bytes</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="font-semibold text-slate-800">Layer 3: Linear(32, 2) Classification Head</span>
                <span className="font-mono text-cyan-800 font-bold">264 bytes</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-cyan-50 border border-cyan-200">
                <span className="font-bold text-cyan-900">Total Uncompressed Weight Payload</span>
                <span className="font-mono text-cyan-900 font-extrabold">12,168 bytes (~11.88 KB)</span>
              </div>
            </div>

            <p className="text-slate-600 text-[11px] leading-relaxed">
              Because the neural network is tightly parameterized for tabular heart-disease risk prediction, communication consumption is negligible even over cellular or standard hospital internet backbones.
            </p>
          </div>
        </FrostedCard>

        <FrostedCard
          title="Bandwidth vs Centralized Raw Pooling"
          subtitle="Comparative analysis of data transmission tradeoffs"
          badge="Bandwidth Tradeoff"
        >
          <div className="space-y-4 text-xs">
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Centralized Data Transmission:</span>
                <strong className="text-slate-900 font-mono">0.467 MB</strong>
              </div>
              <p className="text-[11px] text-slate-500">
                Transferring 12,000 raw CSV rows once uses 0.467 MB, but <strong>destroys patient privacy</strong> and violates HIPAA/GDPR constraints.
              </p>

              <div className="pt-2 border-t border-slate-200 flex justify-between items-center">
                <span className="text-cyan-800 font-bold">Federated Learning (20 Rounds):</span>
                <strong className="text-cyan-900 font-mono font-extrabold text-sm">1.671 MB</strong>
              </div>
              <p className="text-[11px] text-slate-500">
                Consumes just 3.58x the raw data size across the entire training lifecycle while ensuring <strong>zero patient records ever leave hospital firewalls</strong>.
              </p>
            </div>

            <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
              <CheckCircle2 className="size-4 text-emerald-600 shrink-0" />
              <span className="text-[11px]">
                Ideal for continuous online federated learning without hospital network congestion.
              </span>
            </div>
          </div>
        </FrostedCard>
      </div>
    </div>
  );
};
