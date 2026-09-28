import React from 'react';
import {
  Menu,
  ShieldCheck,
  Zap,
  Activity,
  Github,
  Stethoscope,
  ExternalLink,
  Lock,
  Layers
} from 'lucide-react';
import { ScreenId } from './Sidebar';

interface HeaderProps {
  activeScreen: ScreenId;
  onOpenMobile: () => void;
  onQuickScreening: () => void;
}

const SCREEN_TITLES: Record<ScreenId, { title: string; subtitle: string; category: string }> = {
  'command-center': {
    title: 'Command Center & Research Overview',
    subtitle: 'Unified executive telemetry, federated consensus metrics, and integrity benchmarks',
    category: 'Executive'
  },
  'network': {
    title: 'Hospital Topology & Distribution',
    subtitle: 'Multi-institutional node architecture across 6 independent clinical centers',
    category: 'Network'
  },
  'hospital-detail': {
    title: 'Hospital Node Deep-Dive',
    subtitle: 'Granular sample counts, local vs global performance, and clinical distribution',
    category: 'Node Inspection'
  },
  'training': {
    title: 'Federated Training Trajectory',
    subtitle: 'Round-by-round convergence tracking, loss dynamics, and hospital equity gaps',
    category: 'Optimization'
  },
  'non-iid': {
    title: 'Non-IID Label Skew & Dirichlet Study',
    subtitle: 'Impact of statistical heterogeneity on consensus convergence across alpha parameters',
    category: 'Heterogeneity'
  },
  'fedprox': {
    title: 'FedProx Heterogeneity Regularization',
    subtitle: 'Proximal term mu optimization preventing client divergence under clinical imbalance',
    category: 'Optimization'
  },
  'personalization': {
    title: 'Personalization Suite (FedPer & FedBN)',
    subtitle: 'Base-head decoupled architectures achieving peak in-project AUC (0.8650)',
    category: 'Personalization'
  },
  'secure-aggregation': {
    title: 'Secure Aggregation Cryptographic Protocol',
    subtitle: 'Simulated Shamir secret sharing and homomorphic masking preventing server snooping',
    category: 'Security'
  },
  'differential-privacy': {
    title: 'Differential Privacy & Privacy-Utility Tradeoff',
    subtitle: 'Additive Gaussian noise perturbations and empirical utility boundaries',
    category: 'Privacy'
  },
  'attacks': {
    title: 'Adversarial Attacks & Byzantine Defenses',
    subtitle: 'Robustness testing under label-flipping and model poisoning with Multi-Krum & Median',
    category: 'Robustness'
  },
  'risk-screening': {
    title: 'Patient Heart Risk Screening',
    subtitle: 'Interactive diagnostic calculator evaluated against the trained federated MLP model',
    category: 'Clinical Diagnostic'
  },
  'shap': {
    title: 'SHAP Explainability & Feature Importance',
    subtitle: 'Game-theoretic attribution of 13 clinical biomarkers predicting ischemic cardiac events',
    category: 'Explainability'
  },
  'data-exploration': {
    title: 'Clinical Dataset Exploration',
    subtitle: 'Statistical summary of 12,000 synthesized patient records across 6 hospital partitions',
    category: 'Dataset'
  },
  'model-comparison': {
    title: 'Model Benchmarks & State-of-the-Art',
    subtitle: 'Comparative analysis of Centralized, Local-only, FedAvg, FedProx, and Tree Ensembles',
    category: 'Evaluation'
  },
  'communication-cost': {
    title: 'Network Communication Cost & Efficiency',
    subtitle: 'Payload metrics, packet compression ratios, and bandwidth conservation analysis',
    category: 'Efficiency'
  },
  'research-figures': {
    title: 'Publication Figures & Visual Evidence',
    subtitle: 'High-resolution publication charts generated from exact empirical experiment runs',
    category: 'Evidence Gallery'
  }
};

export const Header: React.FC<HeaderProps> = ({
  activeScreen,
  onOpenMobile,
  onQuickScreening
}) => {
  const meta = SCREEN_TITLES[activeScreen] || SCREEN_TITLES['command-center'];

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-18 px-4 sm:px-6 lg:px-8 bg-white/80 backdrop-blur-xl border-b border-slate-200/80 shadow-xs">
      {/* Left: Mobile Toggle & Breadcrumbs */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobile}
          className="p-2 -ml-2 text-slate-600 rounded-xl hover:bg-slate-100 lg:hidden transition-colors"
          aria-label="Open sidebar"
        >
          <Menu className="size-5" />
        </button>

        <div>
          <div className="flex items-center gap-2 text-[11px] font-medium text-slate-400">
            <span>FedCare</span>
            <span>/</span>
            <span className="text-cyan-700 font-semibold">{meta.category}</span>
          </div>
          <h1 className="font-heading font-bold text-slate-900 text-base sm:text-lg leading-tight truncate">
            {meta.title}
          </h1>
        </div>
      </div>

      {/* Right: Telemetry Status Badges & Quick Action */}
      <div className="flex items-center gap-3">
        {/* Status Pills (Desktop) */}
        <div className="hidden xl:flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200/80 text-[11px] font-medium text-emerald-800">
            <span className="size-1.5 rounded-full bg-emerald-500 animate-ping" />
            <ShieldCheck className="size-3.5 text-emerald-600" />
            <span>DP Active (Simulated)</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-50 border border-cyan-200/80 text-[11px] font-medium text-cyan-800">
            <Activity className="size-3.5 text-cyan-600" />
            <span>FedAvg AUC: 0.8493</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 border border-slate-200 text-[11px] font-medium text-slate-700 font-mono">
            <Layers className="size-3.5 text-slate-500" />
            <span>3,042 Params</span>
          </div>
        </div>

        {/* Action Button */}
        {activeScreen !== 'risk-screening' && (
          <button
            onClick={onQuickScreening}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-cyan-700 hover:from-cyan-500 hover:to-cyan-600 text-white text-xs font-semibold shadow-sm shadow-cyan-700/20 hover:shadow-md transition-all"
          >
            <Stethoscope className="size-3.5" />
            <span className="hidden sm:inline">Screen Patient</span>
          </button>
        )}

        {/* GitHub Link */}
        <a
          href="https://github.com/SuhasRam356/FedCare-Privacy-Preserving-Federated-Learning-br-for-Heart-Disease-Prediction"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center size-9 rounded-xl border border-slate-200 bg-white hover:bg-slate-50 text-slate-600 hover:text-slate-900 transition-colors"
          title="View GitHub Repository"
        >
          <Github className="size-4" />
        </a>
      </div>
    </header>
  );
};
