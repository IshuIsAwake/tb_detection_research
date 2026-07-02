import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/design-system";
import { Icon } from "@/components/Shell";
import { ContentsDropdown } from "@/components/Reading";

function Para({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ fontFamily: "var(--font-sans)", fontSize: "var(--text-lead)", color: "var(--ink-2)", lineHeight: "var(--lh-body)", margin: "0 0 var(--sp-4)", maxWidth: "68ch" }}>
      {children}
    </p>
  );
}

export function OverviewPage() {
  const navigate = useNavigate();
  return (
    <main style={{ maxWidth: "var(--content-max)", margin: "0 auto", padding: "0 var(--gutter) var(--sp-9)" }}>
      <header style={{ padding: "var(--sp-9) 0 var(--sp-6)" }}>
        <h1 style={{ fontFamily: "var(--font-serif)", fontSize: "clamp(2.4rem, 5vw, 3.4rem)", fontWeight: "var(--w-semibold)", letterSpacing: "var(--ls-tight)", lineHeight: 1.08, margin: "0 0 var(--sp-5)" }}>
          Detecting tuberculosis in chest X-rays
        </h1>
        <Para>
          Tuberculosis (TB) is a contagious bacterial disease that primarily affects the lungs and remains one
          of the world's leading infectious killers. Early diagnosis is critical for effective treatment, but
          manual interpretation of chest X-rays can be time-consuming and depends heavily on expert radiologists,
          who are not always available in resource-limited settings.
        </Para>
        <Para>
          This research explores the use of deep learning for automated TB detection from chest X-ray images.
          The long-term goal is to design and rigorously evaluate an architecture that achieves state-of-the-art
          performance while remaining reliable, interpretable, and practical for real-world clinical deployment.
          The website documents the project's progress, experiments, and findings as an evolving research log.
        </Para>
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "var(--sp-6)", flexWrap: "wrap" }}>
          <Button variant="primary" onClick={() => navigate("/results")} iconRight={<Icon name="arrow-right" size="1em" />}>Read results &amp; ablation</Button>
          <Button variant="primary" onClick={() => navigate("/about-tb")}>About TB</Button>
        </div>
        <div style={{ marginTop: "var(--sp-6)" }}>
          <ContentsDropdown />
        </div>
      </header>
    </main>
  );
}
