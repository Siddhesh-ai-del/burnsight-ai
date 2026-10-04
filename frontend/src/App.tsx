import { Background } from "./components/Background";
import { Footer } from "./components/Footer";
import { Masthead } from "./components/Masthead";
import { Panel } from "./components/Panel";

export function App() {
  return (
    <div className="shell">
      <Background />
      <Masthead />
      <main className="console">
        <Panel index="01" title="Payload intake">
          <p className="standby">Bay empty</p>
        </Panel>
        <Panel index="02" title="Triage results">
          <p className="standby">Awaiting lot upload</p>
        </Panel>
      </main>
      <Footer />
    </div>
  );
}
