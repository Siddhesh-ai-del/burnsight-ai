/* Sticky masthead — brand, kicker, tagline, system LED.
   Phase 3: the whole bar is a GlassSurface (material mode) — the lens
   bends the live telemetry background via backdrop displacement while
   the content inside renders raw and crisp. */
import { GlassSurface } from "@/components/glass/GlassSurface";

export function Masthead() {
  return (
    <header className="masthead">
      <GlassSurface name="header" className="masthead-glass">
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
      </GlassSurface>
    </header>
  );
}
