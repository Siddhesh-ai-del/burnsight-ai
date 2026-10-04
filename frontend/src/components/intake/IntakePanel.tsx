/* Panel 01 — payload intake: ObsidianUI file dropzone, primary action,
   miss-cost ratio slider (re-runs triage server-side on commit),
   persistent status alert. */
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { GlassSurface } from "@/components/glass/GlassSurface";
import FileInput from "@/components/block/file-input";
import { Terminal } from "lucide-react";

interface IntakePanelProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  exponent: number;
  onExponentChange: (exponent: number) => void;
  onExponentCommit: (exponent: number) => void;
  pStarText: string;
  busy: boolean;
  hasResults: boolean;
  status: string | null;
  onAnalyze: () => void;
}

export function IntakePanel({
  file,
  onFileChange,
  exponent,
  onExponentChange,
  onExponentCommit,
  pStarText,
  busy,
  hasResults,
  status,
  onAnalyze,
}: IntakePanelProps) {
  return (
    <>
      <FileInput
        accept=".csv,.json"
        maxSizeInMB={5}
        onFileChange={(files) => onFileChange(files?.[0] ?? null)}
      />

      <div className="intake-actions">
        {/* Phase 3: material-mode glass — button inside the lens, crisp,
            backdrop bend of the scale ticks beneath. */}
        <GlassSurface name="button" className="run-glass" radius={8}>
          <Button
            type="button"
            onClick={onAnalyze}
            disabled={!file || busy}
            className="run-btn"
          >
            {busy ? <Spinner className="animate-spin" /> : null}
            {busy ? "Screening…" : hasResults ? "Re-run screening" : "Analyze lot"}
          </Button>
        </GlassSurface>
        <span className="hint num intake-file">
          {file ? file.name : "no lot loaded"}
        </span>
      </div>

      <Separator className="intake-sep" />

      <div className="slider-row">
        <div className="slider-label">
          <span className="slider-name">
            Miss-cost ratio <strong>α/β</strong>
          </span>
          <output className="num slider-alpha">
            α = 10<sup>{exponent}</sup>
          </output>
        </div>
        <Slider
          aria-label="Miss-cost ratio alpha over beta"
          value={[exponent]}
          min={1}
          max={15}
          step={1}
          onValueChange={([value]) => onExponentChange(value)}
          onValueCommit={([value]) => onExponentCommit(value)}
          className="alpha-slider"
        />
        <span className="hint num">p* = {pStarText}</span>
      </div>

      {status ? (
        <Alert className="intake-status">
          <Terminal />
          <AlertTitle>Screening log</AlertTitle>
          <AlertDescription className="num">{status}</AlertDescription>
        </Alert>
      ) : null}
    </>
  );
}
