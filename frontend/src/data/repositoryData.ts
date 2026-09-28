// repositoryData.ts - Exact repository-backed experiment data & clinical metrics

export interface RoundMetric {
  round: number;
  train_loss: number;
  test_loss: number;
  accuracy: number;
  auc: number;
  hosp_avg_accuracy: number;
  hosp_avg_auc: number;
  hosp_1_auc: number;
  hosp_2_auc: number;
  hosp_3_auc: number;
  hosp_4_auc: number;
  hosp_5_auc: number;
  hosp_6_auc: number;
}

export const FEDAVG_ROUNDS: RoundMetric[] = [
  { round: 1, train_loss: 0.4440, test_loss: 0.4471, accuracy: 0.7954, auc: 0.8356, hosp_avg_accuracy: 0.7696, hosp_avg_auc: 0.8132, hosp_1_auc: 0.8592, hosp_2_auc: 0.8268, hosp_3_auc: 0.7794, hosp_4_auc: 0.8386, hosp_5_auc: 0.8115, hosp_6_auc: 0.7640 },
  { round: 2, train_loss: 0.4225, test_loss: 0.4300, accuracy: 0.8021, auc: 0.8433, hosp_avg_accuracy: 0.7762, hosp_avg_auc: 0.8184, hosp_1_auc: 0.8579, hosp_2_auc: 0.8200, hosp_3_auc: 0.7884, hosp_4_auc: 0.8500, hosp_5_auc: 0.8182, hosp_6_auc: 0.7756 },
  { round: 3, train_loss: 0.4186, test_loss: 0.4262, accuracy: 0.8008, auc: 0.8453, hosp_avg_accuracy: 0.7758, hosp_avg_auc: 0.8197, hosp_1_auc: 0.8593, hosp_2_auc: 0.8186, hosp_3_auc: 0.7894, hosp_4_auc: 0.8530, hosp_5_auc: 0.8184, hosp_6_auc: 0.7797 },
  { round: 4, train_loss: 0.4155, test_loss: 0.4241, accuracy: 0.8008, auc: 0.8460, hosp_avg_accuracy: 0.7758, hosp_avg_auc: 0.8201, hosp_1_auc: 0.8598, hosp_2_auc: 0.8186, hosp_3_auc: 0.7884, hosp_4_auc: 0.8535, hosp_5_auc: 0.8182, hosp_6_auc: 0.7821 },
  { round: 5, train_loss: 0.4161, test_loss: 0.4226, accuracy: 0.8017, auc: 0.8470, hosp_avg_accuracy: 0.7771, hosp_avg_auc: 0.8208, hosp_1_auc: 0.8603, hosp_2_auc: 0.8183, hosp_3_auc: 0.7889, hosp_4_auc: 0.8552, hosp_5_auc: 0.8186, hosp_6_auc: 0.7832 },
  { round: 6, train_loss: 0.4146, test_loss: 0.4226, accuracy: 0.8037, auc: 0.8472, hosp_avg_accuracy: 0.7796, hosp_avg_auc: 0.8209, hosp_1_auc: 0.8608, hosp_2_auc: 0.8174, hosp_3_auc: 0.7889, hosp_4_auc: 0.8560, hosp_5_auc: 0.8187, hosp_6_auc: 0.7834 },
  { round: 7, train_loss: 0.4115, test_loss: 0.4220, accuracy: 0.8033, auc: 0.8471, hosp_avg_accuracy: 0.7779, hosp_avg_auc: 0.8210, hosp_1_auc: 0.8615, hosp_2_auc: 0.8171, hosp_3_auc: 0.7887, hosp_4_auc: 0.8558, hosp_5_auc: 0.8187, hosp_6_auc: 0.7843 },
  { round: 8, train_loss: 0.4128, test_loss: 0.4220, accuracy: 0.8025, auc: 0.8472, hosp_avg_accuracy: 0.7771, hosp_avg_auc: 0.8212, hosp_1_auc: 0.8617, hosp_2_auc: 0.8185, hosp_3_auc: 0.7881, hosp_4_auc: 0.8560, hosp_5_auc: 0.8181, hosp_6_auc: 0.7848 },
  { round: 9, train_loss: 0.4123, test_loss: 0.4220, accuracy: 0.8042, auc: 0.8471, hosp_avg_accuracy: 0.7746, hosp_avg_auc: 0.8215, hosp_1_auc: 0.8614, hosp_2_auc: 0.8195, hosp_3_auc: 0.7894, hosp_4_auc: 0.8554, hosp_5_auc: 0.8187, hosp_6_auc: 0.7846 },
  { round: 10, train_loss: 0.4108, test_loss: 0.4218, accuracy: 0.8037, auc: 0.8470, hosp_avg_accuracy: 0.7750, hosp_avg_auc: 0.8216, hosp_1_auc: 0.8616, hosp_2_auc: 0.8204, hosp_3_auc: 0.7898, hosp_4_auc: 0.8555, hosp_5_auc: 0.8186, hosp_6_auc: 0.7840 },
  { round: 11, train_loss: 0.4113, test_loss: 0.4220, accuracy: 0.8037, auc: 0.8474, hosp_avg_accuracy: 0.7754, hosp_avg_auc: 0.8217, hosp_1_auc: 0.8615, hosp_2_auc: 0.8203, hosp_3_auc: 0.7900, hosp_4_auc: 0.8556, hosp_5_auc: 0.8186, hosp_6_auc: 0.7845 },
  { round: 12, train_loss: 0.4076, test_loss: 0.4220, accuracy: 0.8037, auc: 0.8470, hosp_avg_accuracy: 0.7742, hosp_avg_auc: 0.8217, hosp_1_auc: 0.8613, hosp_2_auc: 0.8221, hosp_3_auc: 0.7898, hosp_4_auc: 0.8549, hosp_5_auc: 0.8184, hosp_6_auc: 0.7836 },
  { round: 13, train_loss: 0.4103, test_loss: 0.4218, accuracy: 0.8042, auc: 0.8470, hosp_avg_accuracy: 0.7729, hosp_avg_auc: 0.8215, hosp_1_auc: 0.8612, hosp_2_auc: 0.8212, hosp_3_auc: 0.7896, hosp_4_auc: 0.8557, hosp_5_auc: 0.8178, hosp_6_auc: 0.7836 },
  { round: 14, train_loss: 0.4067, test_loss: 0.4217, accuracy: 0.8025, auc: 0.8467, hosp_avg_accuracy: 0.7729, hosp_avg_auc: 0.8209, hosp_1_auc: 0.8606, hosp_2_auc: 0.8192, hosp_3_auc: 0.7892, hosp_4_auc: 0.8552, hosp_5_auc: 0.8175, hosp_6_auc: 0.7837 },
  { round: 15, train_loss: 0.4071, test_loss: 0.4222, accuracy: 0.8054, auc: 0.8466, hosp_avg_accuracy: 0.7742, hosp_avg_auc: 0.8211, hosp_1_auc: 0.8604, hosp_2_auc: 0.8195, hosp_3_auc: 0.7903, hosp_4_auc: 0.8545, hosp_5_auc: 0.8176, hosp_6_auc: 0.7841 },
  { round: 16, train_loss: 0.4046, test_loss: 0.4223, accuracy: 0.8021, auc: 0.8463, hosp_avg_accuracy: 0.7721, hosp_avg_auc: 0.8210, hosp_1_auc: 0.8607, hosp_2_auc: 0.8199, hosp_3_auc: 0.7892, hosp_4_auc: 0.8546, hosp_5_auc: 0.8173, hosp_6_auc: 0.7842 },
  { round: 17, train_loss: 0.4099, test_loss: 0.4221, accuracy: 0.8021, auc: 0.8466, hosp_avg_accuracy: 0.7725, hosp_avg_auc: 0.8211, hosp_1_auc: 0.8597, hosp_2_auc: 0.8210, hosp_3_auc: 0.7897, hosp_4_auc: 0.8545, hosp_5_auc: 0.8175, hosp_6_auc: 0.7841 },
  { round: 18, train_loss: 0.4097, test_loss: 0.4222, accuracy: 0.8025, auc: 0.8466, hosp_avg_accuracy: 0.7729, hosp_avg_auc: 0.8209, hosp_1_auc: 0.8600, hosp_2_auc: 0.8199, hosp_3_auc: 0.7898, hosp_4_auc: 0.8550, hosp_5_auc: 0.8168, hosp_6_auc: 0.7842 },
  { round: 19, train_loss: 0.4064, test_loss: 0.4233, accuracy: 0.8017, auc: 0.8462, hosp_avg_accuracy: 0.7738, hosp_avg_auc: 0.8206, hosp_1_auc: 0.8603, hosp_2_auc: 0.8208, hosp_3_auc: 0.7886, hosp_4_auc: 0.8539, hosp_5_auc: 0.8163, hosp_6_auc: 0.7834 },
  { round: 20, train_loss: 0.4087, test_loss: 0.4232, accuracy: 0.8000, auc: 0.8461, hosp_avg_accuracy: 0.7738, hosp_avg_auc: 0.8208, hosp_1_auc: 0.8598, hosp_2_auc: 0.8217, hosp_3_auc: 0.7887, hosp_4_auc: 0.8538, hosp_5_auc: 0.8174, hosp_6_auc: 0.7832 }
];

export const NON_IID_EXPERIMENTS = [
  { config: "IID (Uniform)", type: "iid", alpha: "Uniform", accuracy: 0.8113, auc: 0.8542, worst_auc: 0.8329, equity_gap: 0.0322, note: "Baseline with identical distributions" },
  { config: "Dirichlet (alpha=1.0)", type: "dirichlet", alpha: "1.0 (Mild)", accuracy: 0.8096, auc: 0.8535, worst_auc: 0.7857, equity_gap: 0.1088, note: "Mild statistical heterogeneity" },
  { config: "Dirichlet (alpha=0.5)", type: "dirichlet", alpha: "0.5 (Moderate)", accuracy: 0.8154, auc: 0.8571, worst_auc: 0.6250, equity_gap: 0.3068, note: "Moderate skew; equity gap widens" },
  { config: "Dirichlet (alpha=0.1)", type: "dirichlet", alpha: "0.1 (Severe)", accuracy: 0.7208, auc: 0.7124, worst_auc: 0.6154, equity_gap: 0.3595, note: "Severe pathological non-IID collapse" },
  { config: "Hospital-Native Skew", type: "hospital_native", alpha: "Real Skew", accuracy: 0.8092, auc: 0.8485, worst_auc: 0.7857, equity_gap: 0.0766, note: "Realistic clinical prevalence variance (4.8% - 46.3%)" }
];

export const FEDPROX_EXPERIMENTS = [
  { algorithm: "FedAvg", mu: 0.0, accuracy: 0.8050, auc: 0.8471, worst_auc: 0.7800, equity_gap: 0.0785, h2_before: 0.8204, h2_after: 0.8200 },
  { algorithm: "FedProx", mu: 0.01, accuracy: 0.8042, auc: 0.8476, worst_auc: 0.7855, equity_gap: 0.0732, h2_before: 0.8134, h2_after: 0.8157 },
  { algorithm: "FedAdam", mu: 0.0, accuracy: 0.8025, auc: 0.8456, worst_auc: 0.7798, equity_gap: 0.0812, h2_before: 0.8239, h2_after: 0.8377 },
  { algorithm: "FedYogi", mu: 0.0, accuracy: 0.8029, auc: 0.8454, worst_auc: 0.7782, equity_gap: 0.0790, h2_before: 0.8203, h2_after: 0.8253 },
  { algorithm: "FedNova", mu: 0.0, accuracy: 0.8021, auc: 0.8474, worst_auc: 0.7844, equity_gap: 0.0714, h2_before: 0.8185, h2_after: 0.8177 },
  { algorithm: "SCAFFOLD", mu: 0.0, accuracy: 0.7208, auc: 0.7971, worst_auc: 0.7165, equity_gap: 0.0934, h2_before: 0.8099, h2_after: 0.8237 },
  { algorithm: "FedPer (Personalized)", mu: 0.0, accuracy: 0.8250, auc: 0.8650, worst_auc: 0.8110, equity_gap: 0.0420, h2_before: 0.8206, h2_after: 0.8650 },
  { algorithm: "FedBN (Personalized)", mu: 0.0, accuracy: 0.8180, auc: 0.8570, worst_auc: 0.7980, equity_gap: 0.0510, h2_before: 0.8174, h2_after: 0.8570 }
];

export const ATTACK_DEFENSE_MATRIX = [
  { attack: "None (Clean)", attack_type: "clean", strategy: "FedAvg (Standard)", accuracy: 0.8017, auc: 0.8461, worst_auc: 0.7810, equity_gap: 0.0784, status: "Baseline" },
  { attack: "None (Clean)", attack_type: "clean", strategy: "Trimmed Mean", accuracy: 0.8108, auc: 0.8479, worst_auc: 0.7834, equity_gap: 0.0770, status: "Robust" },
  { attack: "None (Clean)", attack_type: "clean", strategy: "Coordinate Median", accuracy: 0.8000, auc: 0.8474, worst_auc: 0.7838, equity_gap: 0.0755, status: "Robust" },
  { attack: "None (Clean)", attack_type: "clean", strategy: "Multi-Krum", accuracy: 0.7992, auc: 0.8490, worst_auc: 0.7867, equity_gap: 0.0716, status: "Robust" },

  { attack: "Label-Flipping (Hosp 4, 5)", attack_type: "label_flip", strategy: "FedAvg (Standard)", accuracy: 0.7750, auc: 0.8391, worst_auc: 0.7776, equity_gap: 0.0689, status: "Degraded (-0.7% AUC)" },
  { attack: "Label-Flipping (Hosp 4, 5)", attack_type: "label_flip", strategy: "Trimmed Mean", accuracy: 0.7767, auc: 0.8456, worst_auc: 0.7826, equity_gap: 0.0699, status: "Protected (99.7% AUC)" },
  { attack: "Label-Flipping (Hosp 4, 5)", attack_type: "label_flip", strategy: "Coordinate Median", accuracy: 0.7987, auc: 0.8446, worst_auc: 0.7838, equity_gap: 0.0666, status: "Protected (99.6% AUC)" },
  { attack: "Label-Flipping (Hosp 4, 5)", attack_type: "label_flip", strategy: "Multi-Krum", accuracy: 0.7958, auc: 0.8491, worst_auc: 0.7843, equity_gap: 0.0748, status: "Protected (100% AUC)" },

  { attack: "Model Poisoning (Sign-Flip)", attack_type: "model_poison", strategy: "FedAvg (Standard)", accuracy: 0.2792, auc: 0.5000, worst_auc: 0.5000, equity_gap: 0.0000, status: "Catastrophic Collapse (50% AUC)" },
  { attack: "Model Poisoning (Sign-Flip)", attack_type: "model_poison", strategy: "Trimmed Mean", accuracy: 0.2792, auc: 0.5000, worst_auc: 0.5000, equity_gap: 0.0000, status: "Compromised (Needs >2f+1)" },
  { attack: "Model Poisoning (Sign-Flip)", attack_type: "model_poison", strategy: "Coordinate Median", accuracy: 0.7863, auc: 0.8427, worst_auc: 0.7841, equity_gap: 0.0717, status: "Resilient Defense (0.8427 AUC)" },
  { attack: "Model Poisoning (Sign-Flip)", attack_type: "model_poison", strategy: "Multi-Krum", accuracy: 0.7983, auc: 0.8475, worst_auc: 0.7863, equity_gap: 0.0694, status: "Top Byzantine Defense (0.8475 AUC)" }
];

export const DP_SWEEP_DATA = [
  { noise: 0.0, epsilon: "inf", label: "No DP (Functional)", accuracy: 0.8017, auc: 0.8476, worst_auc: 0.7798, regime: "No Privacy" },
  { noise: 1.68, epsilon: "10.0", label: "Weak DP (eps = 10.0)", accuracy: 0.5779, auc: 0.6112, worst_auc: 0.5540, regime: "Weak DP" },
  { noise: 3.36, epsilon: "5.0", label: "Moderate DP (eps = 5.0)", accuracy: 0.6625, auc: 0.5109, worst_auc: 0.5027, regime: "Moderate DP" },
  { noise: 16.78, epsilon: "1.0", label: "Strong DP (eps = 1.0)", accuracy: 0.4746, auc: 0.5052, worst_auc: 0.4837, regime: "Strong DP" },
  { noise: 33.57, epsilon: "0.5", label: "Extreme DP (eps = 0.5)", accuracy: 0.3675, auc: 0.4095, worst_auc: 0.3553, regime: "Extreme DP" }
];

export const HOSPITALS_INFO = [
  { id: 1, name: "General Metropolitan", region: "North Division", samples: 2000, train: 1600, test: 400, prevalence: 0.235, local_auc: 0.8250, fed_auc: 0.8598, diff: "+0.0348", status: "Active Synced" },
  { id: 2, name: "St. Jude Heart Institute", region: "Low-Prevalence Outlier", samples: 2000, train: 1600, test: 400, prevalence: 0.048, local_auc: 0.7980, fed_auc: 0.8217, diff: "+0.0237", status: "Skewed Outlier (4.8%)" },
  { id: 3, name: "Valley Cardiology Care", region: "Suburban Medical Center", samples: 2000, train: 1600, test: 400, prevalence: 0.379, local_auc: 0.8120, fed_auc: 0.7887, diff: "-0.0233", status: "Active Synced" },
  { id: 4, name: "Highland University Clinic", region: "Academic Hospital", samples: 2000, train: 1600, test: 400, prevalence: 0.207, local_auc: 0.8190, fed_auc: 0.8538, diff: "+0.0348", status: "Targeted in Attacks" },
  { id: 5, name: "Beacon Health Complex", region: "High-Prevalence Center", samples: 2000, train: 1600, test: 400, prevalence: 0.463, local_auc: 0.8040, fed_auc: 0.8174, diff: "+0.0134", status: "Targeted in Attacks" },
  { id: 6, name: "Oakridge Regional Hospital", region: "Rural Health Network", samples: 2000, train: 1600, test: 400, prevalence: 0.342, local_auc: 0.8095, fed_auc: 0.7832, diff: "-0.0263", status: "Active Synced" }
];

export const SHAP_FEATURE_IMPORTANCE = [
  { feature: "Chest Pain: Typical Angina", importance: 0.285, category: "Symptom", clinical_note: "Strongest predictor of acute ischemic heart disease" },
  { feature: "Max Heart Rate (thalach)", importance: 0.242, category: "Vital", clinical_note: "Lower exercise-induced heart rate correlates with coronary deficit" },
  { feature: "Age", importance: 0.198, category: "Demographic", clinical_note: "Monotonic progressive baseline cardiac risk factor" },
  { feature: "Resting Blood Pressure", importance: 0.145, category: "Vital", clinical_note: "Hypertensive vascular strain marker" },
  { feature: "Serum Cholesterol", importance: 0.128, category: "Lab", clinical_note: "Atherosclerotic plaque accumulation driver" },
  { feature: "BMI", importance: 0.095, category: "Biometric", clinical_note: "Metabolic syndrome and cardiovascular burden indicator" },
  { feature: "Fasting Glucose", importance: 0.084, category: "Lab", clinical_note: "Diabetic vascular endothelial damage amplifier" },
  { feature: "Smoker History", importance: 0.076, category: "Lifestyle", clinical_note: "Direct endothelial dysfunction & vasoconstriction trigger" },
  { feature: "Family Cardiac History", importance: 0.065, category: "Genetic", clinical_note: "Heritable coronary predisposition indicator" },
  { feature: "Chest Pain: Non-Anginal", importance: 0.052, category: "Symptom", clinical_note: "Atypical presentation needing differential diagnosis" },
  { feature: "Chest Pain: Atypical Angina", importance: 0.048, category: "Symptom", clinical_note: "Moderately weighted pain symptom" },
  { feature: "Sex (Male = 1)", importance: 0.042, category: "Demographic", clinical_note: "Hormonal and statistical risk factor modifier" },
  { feature: "Diabetes Diagnosis", importance: 0.038, category: "History", clinical_note: "Coronary microvascular disease risk driver" }
];

export const MODEL_BENCHMARKS = [
  { name: "Centralized (Pooled)", auc: 0.8480, accuracy: 0.8083, privacy: "None (Raw Data Pooled)", notes: "Theoretical upper bound with zero privacy guarantees" },
  { name: "Local-Only (Average)", auc: 0.8121, accuracy: 0.8071, privacy: "Isolated (Zero Collaboration)", notes: "Hospitals train solely on private data without FL" },
  { name: "FedAvg (Final Round)", auc: 0.8461, accuracy: 0.8000, privacy: "Federated (No Data Shared)", notes: "Standard FL baseline recovering 94.7% of gap" },
  { name: "FedAvg (Peak Round 11)", auc: 0.8474, accuracy: 0.8037, privacy: "Federated (No Data Shared)", notes: "Optimal global checkpoint" },
  { name: "FedProx (mu = 0.01)", auc: 0.8476, accuracy: 0.8042, privacy: "Federated + Proximal Penalty", notes: "Best global robustness against heterogeneity" },
  { name: "FedPer (Personalized)", auc: 0.8650, accuracy: 0.8250, privacy: "Federated Base + Local Head", notes: "SOTA in-project performance via local classification heads" },
  { name: "Soft Voting Tree Ensemble", auc: 0.8310, accuracy: 0.7940, privacy: "Federated Probability Voting", notes: "Locally trained gradient boosted trees averaged at inference" }
];

export const COMMUNICATION_SPECS = {
  paramCount: 3042,
  modelArchitecture: "13 -> 64 -> 32 -> 2 MLP (Float32)",
  singleMessageKb: 11.88,
  roundKb: 142.59,
  totalFlMb: 1.671,
  centralizedDataMb: 0.467,
  compressionRatio: "12x over raw telemetry",
  encryptionPayloadSecAgg: "3.2x with pairwise masking shares"
};

export const RESEARCH_FIGURES = [
  {
    id: "fig1",
    title: "Figure 1: Federated Convergence vs Baselines",
    file: "/figures/figure1_convergence.png",
    fallbackFile: "/figures/figure1_fedavg_convergence.png",
    phase: "Phase 2 Core",
    metric: "AUC 0.8474 Peak",
    summary: "Validation of FedAvg convergence across 20 federation rounds against Centralized (0.8480) and Local-Only (0.8121) benchmarks, proving that 94.7% of the centralized capability gap is recovered without sharing a single raw patient record."
  },
  {
    id: "fig2",
    title: "Figure 2: Non-IID Dirichlet Skew Impact",
    file: "/figures/figure2_non_iid.png",
    fallbackFile: "/figures/figure2_non_iid_impact.png",
    phase: "Phase 3 Heterogeneity",
    metric: "Alpha = 0.1 (15% AUC drop)",
    summary: "Systematic evaluation of Dirichlet concentration parameter alpha from 1.0 (mild) down to 0.1 (pathological). Demonstrates how severe label distribution skew degrades federated consensus and dramatically increases the hospital equity gap."
  },
  {
    id: "fig3",
    title: "Figure 3: FedProx Heterogeneity Optimization",
    file: "/figures/figure3_fedprox.png",
    fallbackFile: "/figures/figure3_fedprox_vs_fedavg.png",
    phase: "Phase 3 Algorithms",
    metric: "mu = 0.01 Optimal",
    summary: "Investigation of proximal term mu. Small values (0.01) regularize local client updates toward the global model, stabilizing convergence on Hospital 2 (outlier hospital with 4.8% prevalence) while excessive mu (1.0) over-penalizes."
  },
  {
    id: "fig4",
    title: "Figure 4: Adversarial Attacks & Byzantine Defenses",
    file: "/figures/figure4_attacks.png",
    fallbackFile: "/figures/figure4_attacks_and_defenses.png",
    phase: "Phase 4 Security",
    metric: "Multi-Krum & Median Resilience",
    summary: "Trajectory of FedAvg under malicious label-flipping and sign-flip model poisoning by Hospitals 4 and 5. Standard FedAvg collapses to 50% random guessing under model poisoning, while Coordinate Median and Multi-Krum preserve 0.8475 AUC."
  },
  {
    id: "fig5",
    title: "Figure 5: Differential Privacy & Privacy-Utility Tradeoff",
    file: "/figures/figure5_privacy.png",
    fallbackFile: "/figures/figure5_privacy_utility.png",
    phase: "Phase 4 Privacy",
    metric: "Epsilon vs Utility Boundary",
    summary: "Demonstrates additive Gaussian noise injection. Explores the tradeoff frontier between theoretical mathematical privacy budget (epsilon) and clinical utility, highlighting the challenge of high noise in critical diagnostic tasks."
  }
];
