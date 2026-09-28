import React, { useState } from 'react';
import { Image as ImageIcon, ZoomIn, X, Download, FileText, CheckCircle2 } from 'lucide-react';
import { FrostedCard } from '../components/FrostedCard';
import { RESEARCH_FIGURES } from '../data/repositoryData';

export const ResearchFiguresView: React.FC = () => {
  const [selectedFigure, setSelectedFigure] = useState<any | null>(null);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between p-4 rounded-2xl bg-white/80 border border-slate-200/80 backdrop-blur-xl shadow-xs">
        <div>
          <h3 className="font-heading font-bold text-slate-900 text-sm">Publication Figure Gallery</h3>
          <p className="text-xs text-slate-500">
            5 high-resolution analytical figures generated from exact repository experiment runs
          </p>
        </div>
        <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-full bg-cyan-100 text-cyan-800">
          5 Publication Plots
        </span>
      </div>

      {/* Figures Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {RESEARCH_FIGURES.map(fig => (
          <FrostedCard
            key={fig.id}
            title={fig.title}
            subtitle={fig.phase}
            badge={fig.metric}
            badgeColor="bg-cyan-100 text-cyan-800"
            className="flex flex-col justify-between"
          >
            <div className="space-y-4">
              <div
                onClick={() => setSelectedFigure(fig)}
                className="group relative rounded-xl border border-slate-200 overflow-hidden bg-slate-100/50 cursor-pointer aspect-video flex items-center justify-center"
              >
                <img
                  src={fig.file}
                  alt={fig.title}
                  className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-300"
                  onError={(e: any) => {
                    if (fig.fallbackFile && e.target.src !== fig.fallbackFile) {
                      e.target.src = fig.fallbackFile;
                    }
                  }}
                />
                <div className="absolute inset-0 bg-slate-900/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-white text-xs font-semibold backdrop-blur-xs">
                  <ZoomIn className="size-5" />
                  <span>Click to Expand</span>
                </div>
              </div>

              <p className="text-xs text-slate-600 leading-relaxed">
                {fig.summary}
              </p>
            </div>
          </FrostedCard>
        ))}
      </div>

      {/* Fullscreen Lightbox Modal */}
      {selectedFigure && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-slate-900/80 backdrop-blur-md"
          onClick={() => setSelectedFigure(null)}
        >
          <div
            className="relative max-w-5xl w-full max-h-[90vh] bg-white rounded-3xl border border-slate-200 shadow-2xl p-6 overflow-y-auto space-y-4"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div>
                <span className="text-xs font-bold text-cyan-800 uppercase tracking-wider">{selectedFigure.phase}</span>
                <h3 className="font-heading font-extrabold text-slate-900 text-lg sm:text-xl">
                  {selectedFigure.title}
                </h3>
              </div>
              <button
                onClick={() => setSelectedFigure(null)}
                className="p-2 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-700 transition-colors"
              >
                <X className="size-5" />
              </button>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-2 flex items-center justify-center max-h-[60vh] overflow-hidden">
              <img
                src={selectedFigure.file}
                alt={selectedFigure.title}
                className="max-h-[58vh] w-auto object-contain mx-auto"
                onError={(e: any) => {
                  if (selectedFigure.fallbackFile) {
                    e.target.src = selectedFigure.fallbackFile;
                  }
                }}
              />
            </div>

            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 leading-relaxed">
              <strong className="text-slate-900 block mb-1">Analytical Takeaway:</strong>
              {selectedFigure.summary}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
