import React from "react";

interface ButtonProps {
  children?: React.ReactNode;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  iconLeft?: React.ReactNode;
  iconRight?: React.ReactNode;
  disabled?: boolean;
  href?: string;
  onClick?: React.MouseEventHandler<HTMLButtonElement | HTMLAnchorElement>;
  style?: React.CSSProperties;
  [key: string]: unknown;
}

export function Button({
  children,
  variant = "primary",
  size = "md",
  iconLeft = null,
  iconRight = null,
  disabled = false,
  href,
  onClick,
  style = {},
  ...rest
}: ButtonProps) {
  const sizes: Record<string, React.CSSProperties> = {
    sm: { padding: "0.4rem 0.8rem",  fontSize: "var(--text-sm)",   gap: "0.4rem" },
    md: { padding: "0.6rem 1.15rem", fontSize: "var(--text-body)", gap: "0.5rem" },
    lg: { padding: "0.8rem 1.5rem",  fontSize: "var(--text-lead)", gap: "0.6rem" },
  };

  const variants: Record<string, React.CSSProperties> = {
    primary:   { background: "var(--primary)",  color: "var(--on-primary)", border: "1px solid var(--primary)" },
    secondary: { background: "var(--surface)",  color: "var(--ink)",        border: "1px solid var(--line-2)" },
    ghost:     { background: "transparent",     color: "var(--primary)",    border: "1px solid transparent" },
    danger:    { background: "var(--bad)",      color: "#fff",              border: "1px solid var(--bad)" },
  };

  const base: React.CSSProperties = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "var(--font-sans)",
    fontWeight: "var(--w-semibold)",
    lineHeight: 1,
    borderRadius: "var(--r)",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    transition: "background var(--transition), border-color var(--transition), transform var(--dur-fast) var(--ease), box-shadow var(--transition)",
    textDecoration: "none",
    whiteSpace: "nowrap",
    ...sizes[size],
    ...variants[variant],
    ...style,
  };

  const hoverBg: Record<string, string> = {
    primary:   "var(--primary-strong)",
    secondary: "var(--paper-2)",
    ghost:     "var(--primary-tint)",
    danger:    "#9a3d33",
  };

  const handlers = disabled ? {} : {
    onMouseEnter: (e: React.MouseEvent<HTMLElement>) => {
      (e.currentTarget as HTMLElement).style.background = hoverBg[variant];
      if (variant === "secondary") (e.currentTarget as HTMLElement).style.borderColor = "var(--line-3)";
    },
    onMouseLeave: (e: React.MouseEvent<HTMLElement>) => {
      (e.currentTarget as HTMLElement).style.background = variants[variant].background as string;
      if (variant === "secondary") (e.currentTarget as HTMLElement).style.borderColor = "var(--line-2)";
    },
    onMouseDown: (e: React.MouseEvent<HTMLElement>) => { (e.currentTarget as HTMLElement).style.transform = "translateY(1px)"; },
    onMouseUp:   (e: React.MouseEvent<HTMLElement>) => { (e.currentTarget as HTMLElement).style.transform = "translateY(0)"; },
  };

  if (href) {
    return (
      <a style={base} href={href} onClick={disabled ? undefined : onClick as React.MouseEventHandler<HTMLAnchorElement>} {...handlers} {...rest}>
        {iconLeft}{children}{iconRight}
      </a>
    );
  }

  return (
    <button style={base} onClick={disabled ? undefined : onClick as React.MouseEventHandler<HTMLButtonElement>} disabled={disabled} {...handlers} {...rest}>
      {iconLeft}{children}{iconRight}
    </button>
  );
}
