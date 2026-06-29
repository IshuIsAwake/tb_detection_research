import * as React from "react";

export interface TabItem {
  id: string;
  label: React.ReactNode;
}

export interface TabsProps {
  items: TabItem[];
  /** Controlled active id */
  value?: string;
  /** Initial active id when uncontrolled. @default first item */
  defaultValue?: string;
  onChange?: (id: string) => void;
  style?: React.CSSProperties;
}

/**
 * Segmented control — confidence-threshold switches, view toggles. Mono labels,
 * teal active pill on a sunken track.
 */
export function Tabs(props: TabsProps): JSX.Element;
