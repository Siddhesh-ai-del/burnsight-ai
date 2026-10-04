/* Fixed backdrop: telemetry grid, two slow glows, faint scanlines.
   Exists so the Phase 3 liquid-glass surfaces have real content to
   refract. Purely decorative, aria-hidden, pointer-events off. */
export function Background() {
  return (
    <div className="bg-layer" aria-hidden="true">
      <div className="bg-grid" />
      <div className="bg-glow bg-glow--a" />
      <div className="bg-glow bg-glow--b" />
      <div className="bg-scan" />
    </div>
  );
}
