import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/600.css";
import "./index.css";
import { App } from "./App";

const mount = document.getElementById("root");
if (!mount) throw new Error("#root mount point missing from index.html");

createRoot(mount).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
