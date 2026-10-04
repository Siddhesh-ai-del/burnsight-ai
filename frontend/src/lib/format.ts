/* Value formatters — ported verbatim from the vanilla dashboard. */

/** Alpha = 10^exponent (miss-cost ratio of the MVP-5 loss matrix). */
export function alphaValue(exponent: number): number {
  return Math.pow(10, exponent);
}

/** Escalation threshold p* = 1/(alpha+1), precision grows with alpha. */
export function pStar(alpha: number, exponent: number): string {
  return (1 / (alpha + 1)).toFixed(exponent >= 4 ? 8 : 3);
}

/** Risk readout: 0 stays 0, tiny values go exponential, else 3 decimals. */
export function formatRisk(risk: number): string {
  if (risk === 0) return "0";
  if (risk < 1e-3) return risk.toExponential(1);
  return risk.toFixed(3);
}
