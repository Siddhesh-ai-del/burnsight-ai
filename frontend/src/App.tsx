/* BurnSight console — state + wiring.
   Flow (ported from vanilla app.js): pick lot -> POST /api/triage?alpha=
   -> summary readouts + unit table -> row click opens detail drawer.
   Slider commits re-run triage server-side (alpha is never an echo). */
import { useCallback, useState } from "react";
import { toast } from "sonner";
import { Background } from "@/components/Background";
import { Footer } from "@/components/Footer";
import { Masthead } from "@/components/Masthead";
import { Panel } from "@/components/Panel";
import { ClickSpark } from "@/components/block/click-spark";
import SmoothScroll from "@/components/block/smooth-scroll";
import { IntakePanel } from "@/components/intake/IntakePanel";
import { DetailDrawer } from "@/components/results/DetailDrawer";
import { SummaryStrip } from "@/components/results/SummaryStrip";
import { UnitTable } from "@/components/results/UnitTable";
import { Toaster } from "@/components/ui/sonner";
import type { FlaggedDetail, TriageResponse } from "@/lib/api";
import { runTriage } from "@/lib/api";
import { alphaValue, pStar } from "@/lib/format";

export function App() {
  const [exponent, setExponent] = useState(1);
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<TriageResponse | null>(null);
  const [detail, setDetail] = useState<FlaggedDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const runAnalysis = useCallback(
    async (exp: number, lot: File | null) => {
      if (!lot || busy) return;
      setBusy(true);
      setStatus("Running screen → forecast → triage → SHAP … (≈5 s)");
      try {
        const result = await runTriage(lot, alphaValue(exp));
        setData(result);
        setStatus(
          `Analyzed ${result.n_units} units · escalation threshold p* = ` +
            `${result.escalation_threshold.toExponential(2)}`,
        );
        toast.success(
          `Lot triaged: ${result.counts.GREEN} pass · ` +
            `${result.counts.YELLOW} review · ${result.counts.RED} reject`,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setStatus(`Error: ${message}`);
        toast.error(`Screening failed: ${message}`);
      } finally {
        setBusy(false);
      }
    },
    [busy],
  );

  const handleCommit = useCallback(
    (next: number) => {
      setExponent(next);
      if (data) void runAnalysis(next, file);
    },
    [data, file, runAnalysis],
  );

  const alpha = alphaValue(exponent);

  return (
    <SmoothScroll>
      <div className="shell">
        <Background />
        <Masthead />
        <main className="console">
          {data ? <SummaryStrip data={data} /> : null}

          <Panel index="01" title="Payload intake">
            <IntakePanel
              file={file}
              onFileChange={setFile}
              exponent={exponent}
              onExponentChange={setExponent}
              onExponentCommit={handleCommit}
              pStarText={pStar(alpha, exponent)}
              busy={busy}
              hasResults={data !== null}
              status={status}
              onAnalyze={() => void runAnalysis(exponent, file)}
            />
          </Panel>

          <Panel index="02" title="Triage results" className="panel-results">
            {data ? (
              <UnitTable data={data} onSelect={setDetail} />
            ) : (
              <p className="standby">Awaiting lot upload</p>
            )}
          </Panel>
        </main>
        <Footer />

        <DetailDrawer
          detail={detail}
          auditId={data?.audit_id ?? ""}
          onClose={() => setDetail(null)}
        />
        <Toaster position="top-right" />
        <ClickSpark sparkColor="#5cd9e8" />
      </div>
    </SmoothScroll>
  );
}
