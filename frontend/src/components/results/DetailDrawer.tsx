/* Detail drawer (vaul): decision explanation with reasoning collapsible
   and Trajectory / Explain / Certificate tabs.
   Trajectory tab mounts the recharts plot in Phase 4. */
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
} from "@/components/ui/drawer";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { ChevronDown, FileDown, ScrollText } from "lucide-react";
import { cn } from "@/lib/utils";
import type { FlaggedDetail } from "@/lib/api";

interface DetailDrawerProps {
  detail: FlaggedDetail | null;
  auditId: string;
  onClose: () => void;
}

export function DetailDrawer({ detail, auditId, onClose }: DetailDrawerProps) {
  return (
    <Drawer
      open={detail !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DrawerContent className="detail-drawer">
        {detail ? (
          <>
            <DrawerHeader>
              <DrawerTitle className="detail-title">
                Explain decision —{" "}
                <span className="num">{detail.unit_id}</span>
              </DrawerTitle>
              <DrawerDescription className="detail-sub">
                <Badge className={cn("verdict", `verdict--${detail.color}`)}>
                  {detail.color}
                </Badge>
                <code className="reason-code num">{detail.reason_code}</code>
                <span>{detail.defect_mode}</span>
              </DrawerDescription>
            </DrawerHeader>

            <Collapsible className="detail-collapsible">
              <CollapsibleTrigger className="collapsible-trigger">
                <ChevronDown aria-hidden="true" />
                Reasoning
              </CollapsibleTrigger>
              <CollapsibleContent className="collapsible-body">
                <p>{detail.reasons || "—"}</p>
                <p className="num">
                  driver: {detail.worst_channel} · forecast 168 h ={" "}
                  {detail.predicted_168h.toFixed(2)}
                </p>
              </CollapsibleContent>
            </Collapsible>

            <Tabs defaultValue="trajectory" className="detail-tabs">
              <TabsList>
                <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
                <TabsTrigger value="explain">Explain</TabsTrigger>
                <TabsTrigger value="certificate">Certificate</TabsTrigger>
              </TabsList>

              <TabsContent value="trajectory" className="tab-body">
                <div className="standby">Trajectory plot — stage 4</div>
              </TabsContent>

              <TabsContent value="explain" className="tab-body">
                <p className="tab-note">
                  SHAP attribution — top features driving the decision
                </p>
                <ul className="shap-list">
                  {detail.explanation.map((feature) => {
                    const pct = feature.weight * 100;
                    return (
                      <li key={feature.feature}>
                        <span className="feat">{feature.feature}</span>
                        <span className="bar">
                          <span
                            style={{ width: `${Math.max(pct, 1)}%` }}
                          />
                        </span>
                        <span className="val">
                          {feature.shap_value >= 0 ? "+" : ""}
                          {feature.shap_value.toFixed(2)} ({pct.toFixed(1)}%)
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </TabsContent>

              <TabsContent value="certificate" className="tab-body">
                <Card>
                  <CardHeader>
                    <CardTitle className="cert-title">
                      Screening artifacts
                    </CardTitle>
                    <CardDescription>
                      Signed outputs tied to audit record{" "}
                      <span className="num">{auditId.slice(0, 8)}</span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="cert-actions">
                  <Button asChild type="button">
                    <a
                      href={`/api/certificate/${auditId}/${detail.unit_id}`}
                      download
                    >
                      <FileDown />
                      Certificate (PDF)
                    </a>
                  </Button>
                  <Button asChild type="button" variant="outline">
                    <a
                      href={`/api/audit/${auditId}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ScrollText />
                      Audit record (JSON)
                    </a>
                  </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </>
        ) : null}
      </DrawerContent>
    </Drawer>
  );
}
