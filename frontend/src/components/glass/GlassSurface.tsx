/* Thin wrapper around @samasante/liquid-glass (pinned 0.1.0).
   Locks the optics so app code never touches the library API directly:
   low strength, subtle dispersion/sheen on the dark console, frost: 0 —
   glassmorphism blur is banned by design, so the lens is pure refraction
   (SVG feDisplacementMap via backdrop-filter) + edge light.

   Gates:
   - prefers-reduced-motion renders a plain div (kill-switch), with
     .glass-fallback keeping the surface readable without the lens.
   - DEFAULT IS OFF — Phase 3 FPS-gate decision (option A): measured on a
     real GPU, every surface failed the >=55 fps gate (baseline 60.0, header
     35.8, button 37.0, thumb 35.9, drawer 28.4, all-on 13.9), so the
     shipped default renders .glass-fallback on all four surfaces.
   - ?glass=on re-enables all four for demos/evaluation (refraction is a
     proven capability — see .playwright-mcp/p3-*.png); ?glass=off is a
     no-op kept for stable URLs; ?glass=header,thumb enables only those. */
import { Glass } from "@samasante/liquid-glass";
import {
  useEffect,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

export type GlassSurfaceName = "header" | "button" | "thumb" | "drawer";

/* Instrument optics: thin rim bend, faint chromatic split, cool sheen.
   strength is a 0..1 fraction of the box — kept low (0.07–0.16) so the
   bend reads as optics, not distortion. */
const OPTICS: Record<
  GlassSurfaceName,
  Record<string, number>
> = {
  header: {
    frost: 0,
    strength: 0.07,
    depth: 0.55,
    curvature: 0.25,
    dispersion: 0.16,
    bend: 0.35,
    sheen: 0.5,
    sheenAngle: 90,
    specular: 0.8,
    brightness: 0.04,
  },
  button: {
    frost: 0,
    strength: 0.09,
    depth: 0.7,
    curvature: 0.3,
    dispersion: 0.2,
    bend: 0.4,
    sheen: 0.6,
    specular: 1,
    brightness: 0.06,
  },
  thumb: {
    frost: 0,
    strength: 0.16,
    depth: 0.4,
    curvature: 0.6,
    dispersion: 0.25,
    bend: 0.5,
    sheen: 0.8,
    specular: 1.1,
    brightness: 0.05,
  },
  drawer: {
    frost: 0,
    strength: 0.1,
    depth: 0.6,
    curvature: 0.3,
    dispersion: 0.18,
    bend: 0.45,
    sheen: 0.5,
    specular: 0.9,
    brightness: 0.05,
  },
};

const GLASS_PARAM = new URLSearchParams(window.location.search).get("glass");

export function glassEnabled(name: GlassSurfaceName): boolean {
  if (GLASS_PARAM === null || GLASS_PARAM === "off") return false;
  if (GLASS_PARAM === "on") return true;
  return GLASS_PARAM
    .split(",")
    .map((token) => token.trim())
    .includes(name);
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

type GlassSurfaceProps = {
  name: GlassSurfaceName;
  /** Pass content for material-mode surfaces (header/button/thumb): the
      material path requires children and renders them raw and crisp.
      Drawer (refract) takes no children — its UI rides as a sibling. */
  children?: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Corner radius in px (lens inherits element radius when omitted). */
  radius?: number;
  /** Lens centre as 0..1 fractions (moving-lens patterns). NOTE: a center
      prop routes GlassDOM (in-place bend of children) — material surfaces
      must not set it. */
  center?: { x: number; y: number };
  /** Copy source to refract instead of the live backdrop (weak-backdrop
      surfaces — e.g. the drawer over its dimmed overlay). */
  refract?: ReactNode;
};

export function GlassSurface({
  name,
  children,
  className,
  style,
  radius,
  center,
  refract,
}: GlassSurfaceProps) {
  const reduced = usePrefersReducedMotion();

  if (reduced || !glassEnabled(name)) {
    const classes = [className, "glass-fallback"].filter(Boolean).join(" ");
    return (
      <div className={classes} style={style}>
        {children}
      </div>
    );
  }

  return (
    <Glass
      className={className}
      style={style}
      radius={radius}
      center={center}
      refract={refract}
      optics={OPTICS[name]}
    >
      {children}
    </Glass>
  );
}
