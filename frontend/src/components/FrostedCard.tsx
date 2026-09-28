import React from 'react';

interface FrostedCardProps {
  title?: string;
  subtitle?: string;
  badge?: string;
  badgeColor?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  hoverable?: boolean;
}

export const FrostedCard: React.FC<FrostedCardProps> = ({
  title,
  subtitle,
  badge,
  badgeColor = 'bg-cyan-100 text-cyan-800',
  action,
  children,
  className = '',
  hoverable = false
}) => {
  return (
    <div
      className={`rounded-2xl border border-slate-200/80 bg-white/80 backdrop-blur-xl shadow-frost p-5 sm:p-6 ${
        hoverable ? 'hover:shadow-frost-lg hover:border-cyan-200/90 transition-all duration-200' : ''
      } ${className}`}
    >
      {(title || subtitle || badge || action) && (
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5 pb-3 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              {title && (
                <h3 className="font-heading font-bold text-slate-900 text-base leading-snug">
                  {title}
                </h3>
              )}
              {badge && (
                <span className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${badgeColor}`}>
                  {badge}
                </span>
              )}
            </div>
            {subtitle && (
              <p className="text-xs text-slate-500 mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
};
