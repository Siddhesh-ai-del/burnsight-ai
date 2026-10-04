/* Verdict summary strip: instrument readouts for Green/Yellow/Red,
   batch meta, audit link, and a FlipText announcement per run. */
import FlipText from "@/components/block/flip-text";
import type { TriageResponse } from "@/lib/api";
import { pStar } from "@/lib/format";

const READOUTS = [
  { color: "GREEN", label: "PASS" },
  { color: "YELLOW", label: "REVIEW" },
  { color: "RED", label: "REJECT" },
] as const;

interface SummaryStripProps {
  data: TriageResponse;
}

export function SummaryStrip({ data }: SummaryStripProps) {
  const meta = [
    `${data.n_units} units`,
    `α = ${data.alpha}`,
    `p* = ${pStar(data.alpha, Math.log10(data.alpha))}`,
  ].join(" · ");

  return (
    <section className="summary-strip" aria-label="Triage summary">
      <div className="summary-flip">
        <FlipText
          key={data.audit_id}
          loop={false}
          together
          duration={1.1}
          className="summary-announce"
        >
          LOT TRIAGED
        </FlipText>
        <span className="summary-meta num">{meta}</span>
      </div>

      <div className="summary-readouts">
        {READOUTS.map(({ color, label }) => (
          <div key={color} className={`readout readout--${color}`}>
            <b className="num">{data.counts[color]}</b>
            <span className="readout-key">{color}</span>
            <span className="readout-label">{label}</span>
          </div>
        ))}
      </div>

      <a
        className="summary-audit num"
        href={`/api/audit/${data.audit_id}`}
        target="_blank"
        rel="noopener noreferrer"
      >
        AUDIT {data.audit_id.slice(0, 8)} ↗
      </a>
    </section>
  );
}
