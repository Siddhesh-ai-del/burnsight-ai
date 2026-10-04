/* Unit table: one row per component, verdict badge, driver channel,
   168 h forecast, risk, reason cell. Flagged rows expand into the
   detail drawer. Rendering via JSX (React escapes values — closes the
   vanilla innerHTML XSS, issue #6, by construction). */
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ChevronRight } from "lucide-react";
import type { FlaggedDetail, TriageResponse } from "@/lib/api";
import { formatRisk } from "@/lib/format";

interface UnitTableProps {
  data: TriageResponse;
  onSelect: (detail: FlaggedDetail) => void;
}

export function UnitTable({ data, onSelect }: UnitTableProps) {
  const detailById = new Map(data.flagged.map((d) => [d.unit_id, d]));

  return (
    <div className="table-scroll">
      <Table className="unit-table">
        <TableHeader>
          <TableRow>
            <TableHead>Unit</TableHead>
            <TableHead>Verdict</TableHead>
            <TableHead>Driver channel</TableHead>
            <TableHead className="text-right">168&thinsp;h forecast</TableHead>
            <TableHead className="text-right">
              <Tooltip>
                <TooltipTrigger asChild>
                  <span tabIndex={0} className="tip-head">
                    Risk
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  Posterior escalation probability — units ≥ p* escalate to
                  the loss matrix
                </TooltipContent>
              </Tooltip>
            </TableHead>
            <TableHead>Why</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody id="unitBody">
          {data.units.map((unit) => {
            const detail = detailById.get(unit.unit_id);
            const expandable = Boolean(detail);
            return (
              <TableRow
                key={unit.unit_id}
                className={cn("unit", unit.color, expandable && "unit--flagged")}
                onClick={() => detail && onSelect(detail)}
              >
                <TableCell className="uid">
                  {detail ? (
                    <ChevronRight className="chev" aria-hidden="true" />
                  ) : (
                    <span className="chev-gap" aria-hidden="true" />
                  )}
                  {unit.unit_id}
                </TableCell>
                <TableCell>
                  <Badge className={cn("verdict", `verdict--${unit.color}`)}>
                    {unit.color}
                  </Badge>
                </TableCell>
                <TableCell>{unit.worst_channel}</TableCell>
                <TableCell className="num text-right">
                  {unit.predicted_168h.toFixed(2)}
                </TableCell>
                <TableCell className="num text-right">
                  {formatRisk(unit.risk)}
                </TableCell>
                <TableCell className="why-cell">{unit.reasons || "—"}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
