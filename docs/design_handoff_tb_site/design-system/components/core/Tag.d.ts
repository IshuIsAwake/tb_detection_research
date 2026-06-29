import * as React from "react";

export interface TagProps {
  children: React.ReactNode;
  /** @default "neutral" */
  tone?: "neutral" | "primary" | "muted";
  style?: React.CSSProperties;
}

/**
 * Monospace chip for configs, hyperparameters, capabilities and identifiers
 * — lower-key than Badge. Use for things like "512 @ batch 16", "mosaic",
 * "CC BY 4.0", "YOLOv8n".
 */
export function Tag(props: TagProps): JSX.Element;
