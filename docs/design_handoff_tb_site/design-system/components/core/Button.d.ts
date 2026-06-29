import * as React from "react";

export interface ButtonProps {
  children: React.ReactNode;
  /** Visual style. @default "primary" */
  variant?: "primary" | "secondary" | "ghost" | "danger";
  /** @default "md" */
  size?: "sm" | "md" | "lg";
  /** Optional Lucide icon element rendered before the label */
  iconLeft?: React.ReactNode;
  /** Optional Lucide icon element rendered after the label */
  iconRight?: React.ReactNode;
  disabled?: boolean;
  /** Render as a different element; ignored when `href` is set (renders <a>). @default "button" */
  as?: "button" | "a";
  /** When set, renders an anchor */
  href?: string;
  onClick?: (e: React.MouseEvent) => void;
  style?: React.CSSProperties;
}

/**
 * Action control for the research site — primary actions, nav CTAs and links.
 * @startingPoint section="Core" subtitle="Primary / secondary / ghost / danger buttons" viewport="700x140"
 */
export function Button(props: ButtonProps): JSX.Element;
