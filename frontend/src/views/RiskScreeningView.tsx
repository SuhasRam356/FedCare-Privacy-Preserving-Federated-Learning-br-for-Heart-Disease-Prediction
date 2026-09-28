import React, { useState } from 'react';
import {
  Stethoscope,
  HeartPulse,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Sparkles,
  RotateCcw,
  ShieldCheck,
  ChevronRight,
  Info
} from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { PatientData, PredictionResult, evaluatePatientRisk } from '../utils/riskModel';

const PRESETS: Record<string, PatientData> = {
  'Typical 55yo Patient': {
    age: 55,
    resting_bp: 124,
    cholesterol: 205,
    max_heart_rate: 150,
    bmi: 25.5,
    glucose: 96,
    sex: 1,
    smoker: 0,
    diabetes_history: 0,
    family_history: 0,
    chest_pain: 'asymptomatic'
  },
  'Lower-Risk Athlete (34yo)': {
    age: 34,
    resting_bp: 112,
    cholesterol: 165,
    max_heart_rate: 178,
    bmi: 22.4,
    glucose: 84,
    sex: 0,
    smoker: 0,
    diabetes_history: 0,
    family_history: 0,
    chest_pain: 'asymptomatic'
  },
  'Elevated-Risk Senior (72yo)': {
    age: 72,
    resting_bp: 158,
    cholesterol: 278,
    max_heart_rate: 116,
    bmi: 31.2,
    glucose: 132,
    sex: 1,
    smoker: 1,
    diabetes_history: 1,
    family_history: 1,
    chest_pain: 'typical'
  }
};

export const RiskScreeningView: React.FC = () => {
  const [patient, setPatient] = useState<PatientData>(PRESETS['Typical 55yo Patient']);
  const [result, setResult] = useState<PredictionResult>(evaluatePatientRisk(PRESETS['Typical 55yo Patient']));
  const [isCalculated, setIsCalculated] = useState<boolean>(true);

  const handleCalculate = (currentData = patient) => {
    const res = evaluatePatientRisk(currentData);
    setResult(res);
    setIsCalculated(true);
  };

  const loadPreset = (name: string) => {
    const preset = PRESETS[name];
    setPatient(preset);
    handleCalculate(preset);
  };

  const updateField = (key: keyof PatientData, val: any) => {
    const updated = { ...patient, [key]: val };
    setPatient(updated);
    handleCalculate(updated); // Real-time reactive updates
  };

  return (
    <div className="space-y-6">
      {/* Presets Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-xs">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mr-1 flex items-center gap-1">
            <Sparkles className="size-3 text-cyan-600" /> Presets:
          </span>
          {Object.keys(PRESETS).map(name => (
            <button
              key={name}
              onClick={() => loadPreset(name)}
              className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-cyan-50 hover:text-cyan-800 text-slate-700 text-xs font-medium transition-colors"
            >
              {name}
            </button>
          ))}
        </div>

        <button
          onClick={() => loadPreset('Typical 55yo Patient')}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 transition-colors"
        >
          <RotateCcw className="size-3.5" />
          <span>Reset Form</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Inputs (2 Cols) */}
        <div className="lg:col-span-2 space-y-6">
          <FrostedCard
            title="Patient Clinical Parameters"
            subtitle="Input patient vitals and cardiac history; the federated model evaluates real-time risk"
            badge="13 Biomarkers"
          >
            <div className="space-y-6">
              {/* Group 1: Vitals */}
              <div className="space-y-3">
                <span className="text-xs font-bold uppercase tracking-wider text-cyan-800 block">
                  1. Clinical Vitals & Biometrics
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Age (years)
                    </label>
                    <input
                      type="number"
                      min={18}
                      max={95}
                      value={patient.age}
                      onChange={e => updateField('age', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Normal: 18 - 95</span>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Resting Blood Pressure (mmHg)
                    </label>
                    <input
                      type="number"
                      min={80}
                      max={220}
                      value={patient.resting_bp}
                      onChange={e => updateField('resting_bp', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Optimal: &lt;120 mmHg</span>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Max Heart Rate (bpm)
                    </label>
                    <input
                      type="number"
                      min={60}
                      max={220}
                      value={patient.max_heart_rate}
                      onChange={e => updateField('max_heart_rate', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Exercise target: 120-190</span>
                  </div>
                </div>
              </div>

              {/* Group 2: Laboratory & Lipids */}
              <div className="space-y-3 pt-4 border-t border-slate-100">
                <span className="text-xs font-bold uppercase tracking-wider text-cyan-800 block">
                  2. Laboratory & Metabolic Profile
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Total Cholesterol (mg/dL)
                    </label>
                    <input
                      type="number"
                      min={100}
                      max={550}
                      value={patient.cholesterol}
                      onChange={e => updateField('cholesterol', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Desirable: &lt;200 mg/dL</span>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Fasting Glucose (mg/dL)
                    </label>
                    <input
                      type="number"
                      min={50}
                      max={300}
                      value={patient.glucose}
                      onChange={e => updateField('glucose', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Fasting: 70 - 99 mg/dL</span>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Body Mass Index (BMI)
                    </label>
                    <input
                      type="number"
                      step={0.1}
                      min={15}
                      max={50}
                      value={patient.bmi}
                      onChange={e => updateField('bmi', Number(e.target.value))}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 font-mono focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    />
                    <span className="text-[10px] text-slate-400 mt-1 block">Normal: 18.5 - 24.9</span>
                  </div>
                </div>
              </div>

              {/* Group 3: Symptoms & History */}
              <div className="space-y-3 pt-4 border-t border-slate-100">
                <span className="text-xs font-bold uppercase tracking-wider text-cyan-800 block">
                  3. Symptoms & Cardiac Risk History
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Chest Pain Classification
                    </label>
                    <select
                      value={patient.chest_pain}
                      onChange={e => updateField('chest_pain', e.target.value as any)}
                      className="w-full px-3 py-2 text-xs rounded-xl bg-white border border-slate-200 text-slate-900 focus:outline-none focus:border-cyan-500 focus:ring-2 focus:ring-cyan-100"
                    >
                      <option value="asymptomatic">Asymptomatic (No Chest Pain)</option>
                      <option value="atypical">Atypical Angina</option>
                      <option value="non_anginal">Non-Anginal Discomfort</option>
                      <option value="typical">Typical Angina (Ischemic Pattern)</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-xs font-semibold text-slate-700 block mb-1">
                      Biological Sex
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => updateField('sex', 0)}
                        className={`py-2 text-xs font-semibold rounded-xl border transition-all ${
                          patient.sex === 0 ? 'bg-cyan-600 text-white border-cyan-600' : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                        }`}
                      >
                        Female
                      </button>
                      <button
                        type="button"
                        onClick={() => updateField('sex', 1)}
                        className={`py-2 text-xs font-semibold rounded-xl border transition-all ${
                          patient.sex === 1 ? 'bg-cyan-600 text-white border-cyan-600' : 'bg-white text-slate-700 border-slate-200 hover:bg-slate-50'
                        }`}
                      >
                        Male
                      </button>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                  <label className="flex items-center gap-2 p-2.5 rounded-xl border border-slate-200 bg-white/60 hover:bg-slate-50 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={patient.smoker === 1}
                      onChange={e => updateField('smoker', e.target.checked ? 1 : 0)}
                      className="size-4 text-cyan-600 rounded focus:ring-cyan-500"
                    />
                    <span className="text-xs font-medium text-slate-700">Active Tobacco Smoker</span>
                  </label>

                  <label className="flex items-center gap-2 p-2.5 rounded-xl border border-slate-200 bg-white/60 hover:bg-slate-50 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={patient.diabetes_history === 1}
                      onChange={e => updateField('diabetes_history', e.target.checked ? 1 : 0)}
                      className="size-4 text-cyan-600 rounded focus:ring-cyan-500"
                    />
                    <span className="text-xs font-medium text-slate-700">Type 2 Diabetes</span>
                  </label>

                  <label className="flex items-center gap-2 p-2.5 rounded-xl border border-slate-200 bg-white/60 hover:bg-slate-50 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={patient.family_history === 1}
                      onChange={e => updateField('family_history', e.target.checked ? 1 : 0)}
                      className="size-4 text-cyan-600 rounded focus:ring-cyan-500"
                    />
                    <span className="text-xs font-medium text-slate-700">Family Cardiac History</span>
                  </label>
                </div>
              </div>
            </div>
          </FrostedCard>
        </div>

        {/* Prediction Results Card (1 Col) */}
        <div className="space-y-6">
          <FrostedCard
            title="Real-Time Risk Assessment"
            subtitle="Evaluated against FedCare's 6-hospital consensus MLP"
            badge="Diagnostic Output"
          >
            <div className="space-y-5">
              {/* Risk Level Badge & Score */}
              <div
                className={`p-5 rounded-2xl border text-center space-y-2 ${
                  result.risk_level === 'ELEVATED RISK'
                    ? 'bg-rose-50/80 border-rose-200 text-rose-900'
                    : result.risk_level === 'MODERATE RISK'
                    ? 'bg-amber-50/80 border-amber-200 text-amber-900'
                    : 'bg-emerald-50/80 border-emerald-200 text-emerald-900'
                }`}
              >
                <span className="text-[11px] font-bold uppercase tracking-wider block">
                  Predicted Classification
                </span>
                <h4 className="text-3xl font-heading font-extrabold font-mono">
                  {result.risk_percent}% Risk
                </h4>
                <div className="inline-block px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide bg-white/90 shadow-xs">
                  {result.risk_level}
                </div>
              </div>

              {/* Progress Bar Gauge */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px] text-slate-500 font-medium">
                  <span>Low (&lt;30%)</span>
                  <span>Moderate (30-60%)</span>
                  <span>Elevated (&gt;60%)</span>
                </div>
                <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden p-0.5 border border-slate-200">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      result.risk_level === 'ELEVATED RISK'
                        ? 'bg-gradient-to-r from-amber-500 to-rose-600'
                        : result.risk_level === 'MODERATE RISK'
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                    style={{ width: `${result.risk_percent}%` }}
                  />
                </div>
              </div>

              {/* Clinical Factors Identified */}
              <div className="space-y-2 pt-3 border-t border-slate-100 text-xs">
                <span className="font-bold text-slate-800 uppercase tracking-wider text-[10px] block">
                  Top Contributing Biomarkers:
                </span>
                <div className="space-y-1.5">
                  {result.contributing_factors.map((f, i) => (
                    <div key={i} className="flex justify-between items-center p-2 rounded-lg bg-slate-50 border border-slate-200/70">
                      <span className="font-medium text-slate-700">{f.feature}</span>
                      <span className={`font-mono text-[11px] font-bold ${f.isHigh ? 'text-rose-600' : 'text-emerald-600'}`}>
                        {f.impact}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Clinical Recommendations */}
              <div className="space-y-2 pt-3 border-t border-slate-100 text-xs">
                <span className="font-bold text-slate-800 uppercase tracking-wider text-[10px] block">
                  Actionable Recommendations:
                </span>
                <ul className="space-y-1.5 text-slate-600 text-[11px]">
                  {result.clinical_recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-1.5">
                      <span className="size-1.5 rounded-full bg-cyan-600 shrink-0 mt-1.5" />
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/70 text-[10px] text-slate-500 flex items-start gap-2">
                <Info className="size-3.5 text-slate-400 shrink-0 mt-0.5" />
                <span>
                  Educational research screening demonstration. Not intended as a standalone diagnostic medical device.
                </span>
              </div>
            </div>
          </FrostedCard>
        </div>
      </div>
    </div>
  );
};
