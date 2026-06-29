import * as React from "react";

export interface CalloutProps {
  /** @default "note" */
  kind?: "note" | "decision" | "landmine" | "retraction";
  /** Override the default kind label */
  title?: string;
  children: React.ReactNode;
  style?: React.CSSProperties;
}

/**
 * Tinted aside that encodes the project's honesty motif — neutral notes,
 * settled decisions, landmines, and struck-through retractions.
 * @startingPoint section="Content" subtitle="Note / decision / landmine / retraction block" viewport="700x150"
 */
export function Callout(props: CalloutProps): JSX.Element;
