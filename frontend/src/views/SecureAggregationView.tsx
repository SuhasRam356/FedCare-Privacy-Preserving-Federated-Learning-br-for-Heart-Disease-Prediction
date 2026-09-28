import React from 'react';
import { Shield, Lock, Key, ArrowRight, CheckCircle2, AlertTriangle, EyeOff, Layers } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';

export const SecureAggregationView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Cryptographic Protocol</span>
          <p className="text-xl font-bold font-heading text-cyan-800">Shamir Masking (Sim)</p>
          <p className="text-[11px] text-cyan-600">Bonawitz et al. inspired design</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Threat Model</span>
          <p className="text-xl font-bold font-heading text-slate-800">Honest-but-Curious</p>
          <p className="text-[11px] text-slate-500">Server cannot reconstruct client updates</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Cancellation Property</span>
          <p className="text-xl font-bold font-heading text-emerald-700">Zero-Sum Σ s_ij = 0</p>
          <p className="text-[11px] text-emerald-600">Pairwise masks cancel out at server</p>
        </div>
      </div>

      {/* Protocol Visual Walkthrough */}
      <FrostedCard
        title="Secure Aggregation Protocol Lifecycle"
        subtitle="How local hospital weights are masked before transmission and reconstructed at the server"
        badge="Protocol Pipeline"
      >
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          {/* Step 1 */}
          <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center size-6 rounded-full bg-cyan-600 text-white font-bold text-[11px]">1</span>
              <span className="font-bold text-slate-800">Pairwise Key Agreement</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              Every pair of hospitals (u, v) agrees on a shared pseudorandom seed using Diffie-Hellman key exchange.
            </p>
            <div className="p-2 rounded bg-white border border-slate-200 font-mono text-[10px] text-slate-600">
              s_uv = PRG(shared_key_uv)
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center size-6 rounded-full bg-cyan-600 text-white font-bold text-[11px]">2</span>
              <span className="font-bold text-slate-800">Local Weight Masking</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              Hospital u masks its true model weights w_u by adding and subtracting pairwise noise vectors s_uv.
            </p>
            <div className="p-2 rounded bg-white border border-slate-200 font-mono text-[10px] text-cyan-800">
              y_u = w_u + Σ(v&gt;u) s_uv - Σ(v&lt;u) s_vu
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center size-6 rounded-full bg-cyan-600 text-white font-bold text-[11px]">3</span>
              <span className="font-bold text-slate-800">Server Aggregation</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              The central server receives only masked vectors y_u. It cannot inspect any individual hospital's parameters.
            </p>
            <div className="p-2 rounded bg-white border border-slate-200 font-mono text-[10px] text-slate-600">
              Server sees: y_1, y_2, ..., y_6
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-4 rounded-xl bg-slate-50/80 border border-slate-200/80 space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center size-6 rounded-full bg-emerald-600 text-white font-bold text-[11px]">4</span>
              <span className="font-bold text-slate-800">Zero-Sum Cancellation</span>
            </div>
            <p className="text-slate-600 leading-relaxed text-[11px]">
              Summing all y_u cancels out all pairwise noise vectors (+s_uv - s_uv = 0), leaving only the exact sum of true weights!
            </p>
            <div className="p-2 rounded bg-emerald-50 border border-emerald-200 font-mono text-[10px] text-emerald-800 font-bold">
              Σ y_u = Σ w_u + 0
            </div>
          </div>
        </div>
      </FrostedCard>

      {/* Disclosures & Simulation Notice */}
      <FrostedCard
        title="Implementation Note & Academic Transparency"
        subtitle="Clarifying simulated vs production-grade cryptography"
      >
        <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 text-xs text-amber-900 space-y-2">
          <div className="flex items-center gap-2 font-bold text-amber-900">
            <AlertTriangle className="size-4 text-amber-600" />
            <span>Research Simulation Disclosure</span>
          </div>
          <p className="text-[11px] leading-relaxed text-amber-800">
            In this repository, the Secure Aggregation module (<code>fedcare/secure_aggregation.py</code>) operates as an educational simulation. Because all 6 hospitals execute sequentially in a single Python process, shares exist in process memory. In a commercial deployment, this protocol would execute across physical network nodes using cryptographically authenticated TLS connections and Galois Field (GF) integer arithmetic.
          </p>
        </div>
      </FrostedCard>
    </div>
  );
};
