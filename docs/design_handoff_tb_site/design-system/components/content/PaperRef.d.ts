import * as React from "react";

export interface PaperRefProps {
  /** Paper title */
  title: React.ReactNode;
  /** Authors · venue · year line */
  meta?: string;
  /** External URL (arXiv / project page) */
  link?: string;
  /** Override the displayed link text (defaults to hostname+path) */
  linkLabel?: string;
  /** "Why it matters here" note */
  note?: React.ReactNode;
  /** Marks a must-read / non-negotiable citation (teal accent + tag) */
  important?: boolean;
  /** Read-tracker state */
  read?: boolean;
  /** Controlled toggle (omit for self-managed checkbox) */
  onToggleRead?: (next: boolean) => void;
  style?: React.CSSProperties;
}

/**
 * Literature reading-list row — title, authors/venue, arXiv link, a "why it
 * matters" note and a personal read-tracker checkbox. Backbone of References.
 * @startingPoint section="Content" subtitle="Citation row with read tracker" viewport="700x150"
 */
export function PaperRef(props: PaperRefProps): JSX.Element;
