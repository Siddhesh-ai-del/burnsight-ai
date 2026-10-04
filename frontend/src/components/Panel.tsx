/* Framed instrument panel: numbered header, status LED, hairline rule.
   Reused by every console section (intake, results, detail). */
import type { ReactNode } from "react";

type PanelProps = {
  index: string;
  title: string;
  children: ReactNode;
  className?: string;
};

export function Panel({ index, title, children, className }: PanelProps) {
  const classes = className ? `panel ${className}` : "panel";
  return (
    <section className={classes}>
      <div className="panel-head">
        <span className="panel-idx">{index}</span>
        <h2 className="panel-title">{title}</h2>
        <span className="panel-rule" aria-hidden="true" />
      </div>
      <div className="panel-body">{children}</div>
    </section>
  );
}
