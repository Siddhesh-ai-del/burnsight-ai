/* Console footer: provenance line + repo link. */
export function Footer() {
  return (
    <footer className="console-footer">
      <div className="console-footer-inner">
        <span>
          Synthetic Arrhenius lot · model trained on reference seeds 1–4 · SHAP
          seeded for reproducible explanations
        </span>
        <a
          href="https://github.com/Siddhesh-ai-del/burnsight-ai"
          target="_blank"
          rel="noopener noreferrer"
        >
          github.com/Siddhesh-ai-del/burnsight-ai ↗
        </a>
      </div>
    </footer>
  );
}
