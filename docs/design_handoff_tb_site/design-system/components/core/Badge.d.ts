import * as React from "react";

export interface BadgeProps {
  children: React.ReactNode;
  /** Status/result meaning. @default "neutral" */
  tone?: "good" | "warn" | "bad" | "info" | "primary" | "neutral";
  /** soft = tinted fill, solid = full fill. @default "soft" */
  variant?: "soft" | "solid";
  style?: React.CSSProperties;
}

/**
 * Uppercase status pill — result tags (improvement / ruled out / capped),
 * dataset roles, run state. Color carries meaning; always pair with a word.
 */
export function Badge(props: BadgeProps): JSX.Element;
