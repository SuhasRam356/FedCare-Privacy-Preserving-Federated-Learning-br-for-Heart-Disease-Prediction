import React, { useState } from 'react';
import { BarChart3, Stethoscope, Activity, HeartPulse, Filter } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { SHAP_FEATURE_IMPORTANCE } from '../data/repositoryData';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';

export const ShapView: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', 'Symptom', 'Vital', 'Lab', 'Biometric', 'Lifestyle', 'Demographic', 'History', 'Genetic'];

  const filteredFeatures = selectedCategory === 'All'
    ? SHAP_FEATURE_IMPORTANCE
    : SHAP_FEATURE_IMPORTANCE.filter(f => f.category === selectedCategory);

  const chartData = [...filteredFeatures]
    .sort((a, b) => b.importance - a.importance)
    .map(f => ({
      name: f.feature,
      importance: Number((f.importance * 100).toFixed(1)),
      category: f.category,
      note: f.clinical_note
    }));

  return (
    <div className="space-y-6">
      {/* Category Filter Pills */}
      <div className="flex flex-wrap items-center gap-2 p-2 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-xs">
        <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mr-2 flex items-center gap-1.5 pl-2">
          <Filter className="size-3 text-cyan-600" /> Filter:
        </span>
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              selectedCategory === cat
                ? 'bg-cyan-600 text-white shadow-xs'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Main Bar Chart */}
      <FrostedCard
        title="Global SHAP Feature Attribution (Mean |SHAP| Value)"
        subtitle="Game-theoretic marginal contribution of 13 features across the 6-hospital aggregated model"
        badge="Phase 1 & 7 Explainability"
      >
        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 10, right: 30, left: 140, bottom: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} unit="%" />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 11, fill: '#1e293b', fontWeight: 500 }}
                width={140}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '12px',
                  border: '1px solid #cbd5e1',
                  fontSize: '12px'
                }}
                formatter={(val: any, name: any, item: any) => [
                  `${val}% attribution`,
                  item.payload.note
                ]}
              />
              <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                {chartData.map((entry, index) => {
                  let color = '#0891b2';
                  if (entry.category === 'Symptom') color = '#e11d48';
                  else if (entry.category === 'Vital') color = '#059669';
                  else if (entry.category === 'Lab') color = '#8b5cf6';
                  else if (entry.category === 'Biometric') color = '#d97706';
                  return <Cell key={`cell-${index}`} fill={color} />;
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </FrostedCard>

      {/* Feature Clinical Explanations Grid */}
      <FrostedCard
        title="Biomarker Clinical Attribution Roster"
        subtitle="Pathophysiological justification for each feature's contribution to risk scoring"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {filteredFeatures.map((f, i) => (
            <div key={i} className="p-3.5 rounded-xl border border-slate-200/80 bg-slate-50/60 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-slate-800 truncate">{f.feature}</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-cyan-100 text-cyan-800 font-bold shrink-0">
                  {(f.importance * 100).toFixed(1)}%
                </span>
              </div>
              <span className="text-[10px] uppercase font-semibold text-slate-400 block tracking-wider">
                Category: {f.category}
              </span>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                {f.clinical_note}
              </p>
            </div>
          ))}
        </div>
      </FrostedCard>
    </div>
  );
};
