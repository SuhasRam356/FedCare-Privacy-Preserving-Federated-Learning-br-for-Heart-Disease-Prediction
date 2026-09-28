import React, { useState } from 'react';
import { Sidebar, ScreenId } from './components/Sidebar';
import { Header } from './components/Header';

// 15 View Components
import { CommandCenterView } from './views/CommandCenterView';
import { HospitalNetworkView } from './views/HospitalNetworkView';
import { HospitalDetailView } from './views/HospitalDetailView';
import { TrainingTrajectoryView } from './views/TrainingTrajectoryView';
import { NonIidView } from './views/NonIidView';
import { FedProxView } from './views/FedProxView';
import { PersonalizationView } from './views/PersonalizationView';
import { SecureAggregationView } from './views/SecureAggregationView';
import { DifferentialPrivacyView } from './views/DifferentialPrivacyView';
import { AttacksDefensesView } from './views/AttacksDefensesView';
import { RiskScreeningView } from './views/RiskScreeningView';
import { ShapView } from './views/ShapView';
import { DataExplorationView } from './views/DataExplorationView';
import { ModelComparisonView } from './views/ModelComparisonView';
import { CommunicationCostView } from './views/CommunicationCostView';
import { ResearchFiguresView } from './views/ResearchFiguresView';

export default function App() {
  const [activeScreen, setActiveScreen] = useState<ScreenId>('command-center');
  const [isMobileOpen, setIsMobileOpen] = useState<boolean>(false);

  const renderActiveView = () => {
    switch (activeScreen) {
      case 'command-center':
        return <CommandCenterView onNavigate={setActiveScreen} />;
      case 'network':
        return <HospitalNetworkView />;
      case 'hospital-detail':
        return <HospitalDetailView />;
      case 'training':
        return <TrainingTrajectoryView />;
      case 'non-iid':
        return <NonIidView />;
      case 'fedprox':
        return <FedProxView />;
      case 'personalization':
        return <PersonalizationView />;
      case 'secure-aggregation':
        return <SecureAggregationView />;
      case 'differential-privacy':
        return <DifferentialPrivacyView />;
      case 'attacks':
        return <AttacksDefensesView />;
      case 'risk-screening':
        return <RiskScreeningView />;
      case 'shap':
        return <ShapView />;
      case 'data-exploration':
        return <DataExplorationView />;
      case 'model-comparison':
        return <ModelComparisonView />;
      case 'communication-cost':
        return <CommunicationCostView />;
      case 'research-figures':
        return <ResearchFiguresView />;
      default:
        return <CommandCenterView onNavigate={setActiveScreen} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 flex">
      {/* Grouped Sidebar Navigation */}
      <Sidebar
        activeScreen={activeScreen}
        onSelectScreen={setActiveScreen}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 lg:pl-72">
        {/* Sticky Top Header */}
        <Header
          activeScreen={activeScreen}
          onOpenMobile={() => setIsMobileOpen(true)}
          onQuickScreening={() => setActiveScreen('risk-screening')}
        />

        {/* Dynamic Page View */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {renderActiveView()}
        </main>

        {/* Global Academic & Privacy Footer */}
        <footer className="mt-auto border-t border-slate-200/80 bg-white/70 backdrop-blur-md px-6 py-4 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-heading font-bold text-slate-700">FedCare</span>
            <span>·</span>
            <span>Privacy-Preserving Federated Learning for Heart Disease Prediction</span>
          </div>
          <div className="flex items-center gap-4 text-[11px] text-slate-400 font-mono">
            <span>6 Institutional Nodes</span>
            <span>•</span>
            <span>20 Rounds FedAvg</span>
            <span>•</span>
            <span>Readable Frost v2.0</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
