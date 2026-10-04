import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./index.css";

const mount = document.getElementById("root");
if (!mount) throw new Error("#root mount point missing from index.html");

createRoot(mount).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
