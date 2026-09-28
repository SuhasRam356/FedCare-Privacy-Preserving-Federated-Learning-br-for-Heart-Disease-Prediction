import React, { useState } from 'react';
import {
  LayoutDashboard,
  Network,
  Activity,
  GitFork,
  Sliders,
  Sparkles,
  Shield,
  EyeOff,
  Skull,
  Stethoscope,
  BarChart3,
  Database,
  GitCompare,
  Wifi,
  Image as ImageIcon,
  Building2,
  ChevronDown,
  Search,
  CheckCircle2,
  ShieldAlert
} from 'lucide-react';

export type ScreenId =
  | 'command-center'
  | 'network'
  | 'hospital-detail'
  | 'training'
  | 'non-iid'
  | 'fedprox'
  | 'personalization'
  | 'secure-aggregation'
  | 'differential-privacy'
  | 'attacks'
  | 'risk-screening'
  | 'shap'
  | 'data-exploration'
  | 'model-comparison'
  | 'communication-cost'
  | 'research-figures';

interface NavItem {
  id: ScreenId;
  label: string;
  icon: React.ElementType;
  badge?: string;
  badgeColor?: string;
}

interface NavGroup {
  groupName: string;
  items: NavItem[];
}

interface SidebarProps {
  activeScreen: ScreenId;
  onSelectScreen: (id: ScreenId) => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
}

export const NAV_GROUPS: NavGroup[] = [
  {
    groupName: 'Executive & Network',
    items: [
      { id: 'command-center', label: 'Command Center', icon: LayoutDashboard, badge: 'Overview' },
      { id: 'network', label: 'Hospital Topology', icon: Network, badge: '6 Nodes' },
      { id: 'hospital-detail', label: 'Hospital Deep-Dive', icon: Building2 }
    ]
  },
  {
    groupName: 'Federated Optimization',
    items: [
      { id: 'training', label: 'Training Trajectory', icon: Activity, badge: '20 Rnds' },
      { id: 'non-iid', label: 'Non-IID Skew Study', icon: GitFork, badge: 'Dirichlet' },
      { id: 'fedprox', label: 'FedProx Heterogeneity', icon: Sliders, badge: 'μ-Sweep' },
      { id: 'personalization', label: 'Personalization Suite', icon: Sparkles, badge: 'FedPer SOTA', badgeColor: 'bg-cyan-100 text-cyan-800' }
    ]
  },
  {
    groupName: 'Privacy & Security',
    items: [
      { id: 'secure-aggregation', label: 'Secure Aggregation', icon: Shield, badge: 'SecAgg' },
      { id: 'differential-privacy', label: 'Differential Privacy', icon: EyeOff, badge: 'DP-SGD' },
      { id: 'attacks', label: 'Attacks & Byzantine', icon: Skull, badge: 'Multi-Krum', badgeColor: 'bg-rose-100 text-rose-800' }
    ]
  },
  {
    groupName: 'Clinical & Diagnostics',
    items: [
      { id: 'risk-screening', label: 'Patient Risk Screening', icon: Stethoscope, badge: 'Live Demo', badgeColor: 'bg-emerald-100 text-emerald-800' },
      { id: 'shap', label: 'SHAP Explainability', icon: BarChart3, badge: '13 Feats' },
      { id: 'data-exploration', label: 'Clinical Dataset', icon: Database, badge: '12k Pts' }
    ]
  },
  {
    groupName: 'Benchmarks & Evidence',
    items: [
      { id: 'model-comparison', label: 'Model Benchmarks', icon: GitCompare, badge: 'SOTA' },
      { id: 'communication-cost', label: 'Network Bandwidth', icon: Wifi, badge: '26.8 KB' },
      { id: 'research-figures', label: 'Publication Figures', icon: ImageIcon, badge: '5 Figs' }
    ]
  }
];

export const Sidebar: React.FC<SidebarProps> = ({
  activeScreen,
  onSelectScreen,
  isMobileOpen,
  onCloseMobile
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});

  const toggleGroup = (name: string) => {
    setCollapsedGroups(prev => ({ ...prev, [name]: !prev[name] }));
  };

  const filteredGroups = NAV_GROUPS.map(group => ({
    ...group,
    items: group.items.filter(item =>
      item.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
      group.groupName.toLowerCase().includes(searchQuery.toLowerCase())
    )
  })).filter(group => group.items.length > 0);

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/30 backdrop-blur-sm lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-72 flex flex-col bg-white/90 backdrop-blur-2xl border-r border-slate-200/80 shadow-frost transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* App Brand Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200/80">
          <div className="flex items-center gap-3">
            <div className="relative flex items-center justify-center size-10 rounded-xl bg-gradient-to-br from-cyan-500 to-cyan-700 text-white shadow-md shadow-cyan-600/20">
              <Activity className="size-5 animate-pulse" />
              <div className="absolute -top-1 -right-1 size-3 rounded-full bg-emerald-500 ring-2 ring-white" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-heading font-extrabold text-lg text-slate-900 tracking-tight">FedCare</span>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded-full bg-cyan-100 text-cyan-800">
                  Frost
                </span>
              </div>
              <p className="text-[11px] font-medium text-slate-500 truncate">
                Privacy-Preserving Federated AI
              </p>
            </div>
          </div>
        </div>

        {/* Search Bar */}
        <div className="p-3 border-b border-slate-100">
          <div className="relative">
            <Search className="absolute left-3 top-2.5 size-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Quick search 15 screens..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 text-xs rounded-xl bg-slate-50 border border-slate-200 text-slate-800 placeholder-slate-400 focus:outline-none focus:border-cyan-500 focus:bg-white focus:ring-2 focus:ring-cyan-100 transition-all"
            />
          </div>
        </div>

        {/* Grouped Navigation List */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
          {filteredGroups.map(group => {
            const isCollapsed = collapsedGroups[group.groupName];
            return (
              <div key={group.groupName} className="space-y-1">
                <button
                  onClick={() => toggleGroup(group.groupName)}
                  className="flex items-center justify-between w-full px-2 py-1 text-[11px] font-bold uppercase tracking-wider text-slate-400 hover:text-slate-700 transition-colors"
                >
                  <span>{group.groupName}</span>
                  <ChevronDown
                    className={`size-3 transition-transform duration-200 ${
                      isCollapsed ? '-rotate-90' : 'rotate-0'
                    }`}
                  />
                </button>

                {!isCollapsed && (
                  <div className="space-y-0.5">
                    {group.items.map(item => {
                      const Icon = item.icon;
                      const isActive = activeScreen === item.id;
                      return (
                        <button
                          key={item.id}
                          onClick={() => {
                            onSelectScreen(item.id);
                            onCloseMobile();
                          }}
                          className={`group flex items-center justify-between w-full px-3 py-2 text-xs font-medium rounded-xl transition-all ${
                            isActive
                              ? 'bg-gradient-to-r from-cyan-600 to-cyan-700 text-white shadow-sm shadow-cyan-700/25 font-semibold'
                              : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/70'
                          }`}
                        >
                          <div className="flex items-center gap-2.5 truncate">
                            <Icon
                              className={`size-4 shrink-0 transition-transform group-hover:scale-110 ${
                                isActive ? 'text-white' : 'text-slate-400 group-hover:text-cyan-600'
                              }`}
                            />
                            <span className="truncate">{item.label}</span>
                          </div>

                          {item.badge && (
                            <span
                              className={`text-[10px] px-1.5 py-0.5 rounded-md font-medium shrink-0 ${
                                isActive
                                  ? 'bg-white/20 text-white'
                                  : item.badgeColor || 'bg-slate-100 text-slate-600'
                              }`}
                            >
                              {item.badge}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* System Health / Footer Card */}
        <div className="p-3 border-t border-slate-200/80 bg-slate-50/60">
          <div className="p-2.5 rounded-xl border border-slate-200/70 bg-white/70 shadow-xs space-y-2">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-700">
                <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
                Network Synchronized
              </span>
              <span className="text-[10px] font-mono text-emerald-700 bg-emerald-50 px-1 rounded">
                6/6 Active
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>AUC Consensus:</span>
              <span className="font-semibold text-cyan-700 font-mono">0.8493</span>
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>Academic Integrity:</span>
              <span className="inline-flex items-center gap-0.5 text-emerald-600 font-medium text-[10px]">
                <CheckCircle2 className="size-3" /> Real Validated
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
