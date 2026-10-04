/* Sticky masthead — brand, kicker, tagline, system LED.
   Phase 3 wraps the bar in the liquid-glass surface. */
export function Masthead() {
  return (
    <header className="masthead">
      <div className="masthead-inner">
        <div>
          <span className="kicker">ISRO SIH 2026 · PS 26170</span>
          <h1 className="brand">
            BurnSight <span className="brand-accent">AI</span>
          </h1>
          <p className="tagline">
            AI-driven anomaly detection in component burn-in &amp; screening —
            upload an early-lot sample (0/24/96&nbsp;h), get a screened,
            168&nbsp;h-forecasted, risk-triaged verdict.
          </p>
        </div>
        <div className="mast-status" role="status">
          <span className="led" aria-hidden="true" />
          SYS · STANDBY
        </div>
      </div>
    </header>
  );
}
