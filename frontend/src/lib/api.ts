/* Typed client for the BurnSight triage API (port of the vanilla app.js
   fetch flow — same endpoints, no backend changes). */

export type VerdictColor = "GREEN" | "YELLOW" | "RED";

export interface UnitRow {
  unit_id: string;
  color: VerdictColor;
  worst_channel: string;
  predicted_168h: number;
  headroom: number;
  risk: number;
  reasons: string;
}

export interface ShapFeature {
  feature: string;
  weight: number;
  shap_value: number;
}

export interface Trajectory {
  channel: string;
  hours: number[];
  forecast: number[];
  measured: Record<string, number>;
  predicted_168h: number;
  limits: { lsl: number; usl: number };
}

export interface FlaggedDetail {
  unit_id: string;
  color: VerdictColor;
  reason_code: string;
  defect_mode: string;
  explanation: ShapFeature[];
  worst_channel: string;
  predicted_168h: number;
  reasons: string;
  chart: Trajectory;
}

export interface TriageResponse {
  n_units: number;
  counts: Record<VerdictColor, number>;
  alpha: number;
  beta: number;
  escalation_threshold: number;
  audit_id: string;
  input_digest: string;
  units: UnitRow[];
  flagged: FlaggedDetail[];
}

export async function runTriage(
  file: File,
  alpha: number,
): Promise<TriageResponse> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`/api/triage?alpha=${alpha}`, {
    method: "POST",
    body,
  });
  const data: unknown = await response.json();
  if (!response.ok) {
    const detail = (data as { detail?: { message?: string } }).detail;
    const fallback = JSON.stringify(data);
    throw new Error(detail?.message || fallback);
  }
  return data as TriageResponse;
}
