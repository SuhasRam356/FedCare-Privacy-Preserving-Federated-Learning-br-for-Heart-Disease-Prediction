import React from 'react';
import { Database, Filter, Layers, BarChart2, Table } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';

const FEATURE_STATS = [
  { feature: "age", label: "Patient Age", range: "29.0 - 77.0", mean: "54.4", std: "9.1", type: "Continuous", unit: "years" },
  { feature: "resting_bp", label: "Resting Blood Pressure", range: "94.0 - 200.0", mean: "131.7", std: "17.6", type: "Continuous", unit: "mmHg" },
  { feature: "cholesterol", label: "Serum Cholesterol", range: "126.0 - 564.0", mean: "246.3", std: "51.8", type: "Continuous", unit: "mg/dL" },
  { feature: "max_heart_rate", label: "Maximum Heart Rate", range: "71.0 - 202.0", mean: "149.6", std: "22.9", type: "Continuous", unit: "bpm" },
  { feature: "bmi", label: "Body Mass Index", range: "18.0 - 40.0", mean: "29.0", std: "6.3", type: "Continuous", unit: "kg/m²" },
  { feature: "glucose", label: "Fasting Blood Sugar", range: "70.0 - 200.0", mean: "135.0", std: "37.5", type: "Continuous", unit: "mg/dL" },
  { feature: "sex", label: "Biological Sex", range: "0.0 or 1.0", mean: "0.68", std: "0.47", type: "Binary", unit: "0=F, 1=M" },
  { feature: "smoker", label: "Tobacco Smoker", range: "0.0 or 1.0", mean: "0.50", std: "0.50", type: "Binary", unit: "0=No, 1=Yes" },
  { feature: "diabetes_history", label: "Diabetes History", range: "0.0 or 1.0", mean: "0.49", std: "0.50", type: "Binary", unit: "0=No, 1=Yes" },
  { feature: "family_history", label: "Family Cardiac History", range: "0.0 or 1.0", mean: "0.50", std: "0.50", type: "Binary", unit: "0=No, 1=Yes" },
  { feature: "cp_atypical_angina", label: "Atypical Angina Chest Pain", range: "0.0 or 1.0", mean: "0.50", std: "0.50", type: "Binary", unit: "One-hot" },
  { feature: "cp_non_anginal", label: "Non-Anginal Discomfort", range: "0.0 or 1.0", mean: "0.50", std: "0.50", type: "Binary", unit: "One-hot" },
  { feature: "cp_typical_angina", label: "Typical Ischemic Angina", range: "0.0 or 1.0", mean: "0.50", std: "0.50", type: "Binary", unit: "One-hot" },
];

export const DataExplorationView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Total Records</span>
          <p className="text-2xl font-bold font-heading text-slate-900 font-mono">12,000</p>
          <p className="text-[11px] text-slate-500">Across 6 hospital CSVs</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Input Dimension</span>
          <p className="text-2xl font-bold font-heading text-cyan-800 font-mono">13 Features</p>
          <p className="text-[11px] text-cyan-600">Standardized with StandardScaler</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Positive Class Ratio</span>
          <p className="text-2xl font-bold font-heading text-slate-900 font-mono">27.9%</p>
          <p className="text-[11px] text-slate-500">3,348 positive / 8,652 negative</p>
        </div>

        <div className="p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-frost space-y-1">
          <span className="text-xs text-slate-500">Per-Hospital Cohort</span>
          <p className="text-2xl font-bold font-heading text-slate-900 font-mono">2,000 Pts</p>
          <p className="text-[11px] text-slate-500">1,600 train / 400 test split</p>
        </div>
      </div>

      {/* Feature Specification Table */}
      <FrostedCard
        title="13-Biomarker Dataset Schema & Statistical Distributions"
        subtitle="Parameters generated under data/heart/hospital_{1..6}.csv calibrated from the Cleveland cardiac distribution"
        badge="Tabular Specification"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                <th className="py-2.5 pr-4">Feature Key</th>
                <th className="py-2.5 px-3">Clinical Name</th>
                <th className="py-2.5 px-3">Data Type</th>
                <th className="py-2.5 px-3">Valid Range</th>
                <th className="py-2.5 px-3">Mean</th>
                <th className="py-2.5 px-3">Std Dev</th>
                <th className="py-2.5 pl-3">Units</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {FEATURE_STATS.map((f, i) => (
                <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                  <td className="py-3 pr-4 font-bold text-slate-900">{f.feature}</td>
                  <td className="py-3 px-3 font-sans font-medium text-slate-700">{f.label}</td>
                  <td className="py-3 px-3">
                    <span className={`px-2 py-0.5 rounded text-[10px] font-sans font-semibold ${
                      f.type === 'Continuous' ? 'bg-cyan-50 text-cyan-800' : 'bg-slate-100 text-slate-700'
                    }`}>
                      {f.type}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-600">{f.range}</td>
                  <td className="py-3 px-3 font-bold text-slate-800">{f.mean}</td>
                  <td className="py-3 px-3 text-slate-500">{f.std}</td>
                  <td className="py-3 pl-3 text-slate-500 font-sans">{f.unit}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </FrostedCard>
    </div>
  );
};
