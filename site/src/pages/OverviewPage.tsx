import React from "react";
import { useNavigate } from "react-router-dom";
import { Button, Badge, Tag } from "@/design-system";
import { Icon } from "@/components/Shell";

function GoalCard({ n, title, body, status }: { n: string; title: string; body: string; status: { tone: "good" | "warn" | "bad" | "info" | "primary" | "neutral"; label: string } }) {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--line)", borderRadius: "var(--r-lg)", padding: "var(--sp-5)", boxShadow: "var(--shadow-sm)", textAlign: "left" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", marginBottom: "0.85rem" }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", fontWeight: "var(--w-semibold)", color: "var(--primary)", border: "1px solid var(--primary-tint-2)", background: "var(--primary-tint)", borderRadius: "var(--r-sm)", padding: "0.15rem 0.45rem" }}>
          {n}
        </span>
        <Badge tone={status.tone}>{status.label}</Badge>
      </div>
      <h3 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h3)", fontWeight: "var(--w-semibold)", margin: "0 0 0.5rem" }}>{title}</h3>
      <p style={{ margin: 0, color: "var(--ink-2)", fontSize: "var(--text-sm)", lineHeight: "var(--lh-body)" }}>{body}</p>
    </div>
  );
}

function PaperSection({ label, title, children }: { label: string; title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: "var(--sp-8)", textAlign: "center" }}>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "var(--text-label)", textTransform: "uppercase", letterSpacing: "var(--ls-label)", color: "var(--primary)", fontWeight: "var(--w-semibold)" }}>{label}</div>
      <h2 style={{ fontFamily: "var(--font-serif)", fontSize: "var(--text-h2)", fontWeight: "var(--w-semibold)", margin: "0.4rem 0 var(--sp-4)", lineHeight: "var(--lh-snug)" }}>{title}</h2>
      {children}
    </section>
  );
}

function Para({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-body)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", margin: "0 auto var(--sp-3)", maxWidth: "62ch" }}>
      {children}
    </p>
  );
}

export function OverviewPage() {
  const navigate = useNavigate();
  return (
    <main style={{ maxWidth: "820px", margin: "0 auto", padding: "0 var(--gutter) var(--sp-9)", textAlign: "center" }}>
      <header style={{ padding: "var(--sp-9) 0 var(--sp-6)" }}>
        <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "clamp(2.4rem, 5vw, 3.4rem)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: 1.08, margin: 0 }}>
          Detecting tuberculosis in chest X-rays
        </h1>
        <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-lead)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", margin: "1.25rem auto 0", maxWidth: "60ch" }}>
          Tuberculosis remains one of the deadliest infectious diseases, yet expert radiologists agree with
          the reference standard only ~69% of the time and trained readers are scarce where the burden is highest.
          This project asks a narrow, testable question: <em>how reliably can an object detector localise active
          TB lesions on a chest radiograph</em> — and what limits that reliability — before committing to a
          purpose-built architecture.
        </p>
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "var(--sp-6)", flexWrap: "wrap", justifyContent: "center" }}>
          <Button variant="primary" onClick={() => navigate("/results")} iconRight={<Icon name="arrow-right" size="1em" />}>Read results &amp; ablation</Button>
          <Button variant="secondary" onClick={() => navigate("/data")}>Dataset research</Button>
          <Button variant="ghost" onClick={() => navigate("/references")}>References</Button>
        </div>
      </header>

      <div style={{ height: "1px", background: "var(--line)", margin: "var(--sp-4) auto 0", maxWidth: "120px" }} />

      <PaperSection label="Abstract" title="What this notebook is">
        <Para>
          A systematic, evidence-first study of TB screening from chest radiographs. Rather than chasing a
          single headline number, each experiment is recorded in order with its decision, its guardrails, and —
          where a reading was later shown wrong — its retraction kept in place. The current phase benchmarks
          off-the-shelf detectors (YOLO, RetinaNet, Faster R-CNN) against a deliberately untuned floor to expose
          where the task actually breaks: recall, class imbalance, and small lesions.
        </Para>
      </PaperSection>

      <PaperSection label="01 · Background" title="Why detection, not classification">
        <Para>
          Image-level classifiers can say <em>a lung looks abnormal</em> but not <em>where</em>, and they degrade
          silently on the co-morbid, non-TB pathology that dominates real screening queues. Localising the lesion
          makes the model auditable by a clinician and separates two failure modes that classifiers conflate:
          missing a true lesion, and firing on unrelated disease.
        </Para>
      </PaperSection>

      <PaperSection label="02 · Aims" title="Two goals">
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "var(--sp-4)", textAlign: "left", marginTop: "var(--sp-2)" }}>
          <GoalCard n="01" title="Benchmark existing architectures" status={{ tone: "good", label: "Active" }}
            body="Establish how off-the-shelf detection architectures actually perform at finding TB lesions — starting from a deliberate, untuned baseline floor, with the evidence kept attached to every decision." />
          <GoalCard n="02" title="Build an optimized architecture" status={{ tone: "neutral", label: "Planned" }}
            body="Use what the benchmark exposes — the recall bottleneck, class imbalance, small lesions — to design a purpose-built detector for the task." />
        </div>
      </PaperSection>

      <PaperSection label="03 · Method" title="Protocol">
        <Para>
          Every run is scored on a sealed test set untouched by training. The optimisation target is active-class
          mAP50 with lesion-level recall, balanced against localisation IoU as a guardrail. Augmentation, sampling
          and resolution are varied one lever at a time, and each lever is validated across multiple seeds before
          it is allowed to change the finalist configuration.
        </Para>
        <div style={{ display: "inline-flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "center", marginTop: "var(--sp-2)" }}>
          <Tag tone="primary">YOLOv8n floor</Tag>
          <Tag>positives-only</Tag>
          <Tag>multi-seed</Tag>
          <Tag tone="primary">VinDr-init + mixup @ 512</Tag>
        </div>
      </PaperSection>
    </main>
  );
}
