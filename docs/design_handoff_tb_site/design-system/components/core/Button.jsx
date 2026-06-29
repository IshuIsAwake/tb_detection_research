import React from "react";

/**
 * Button — the project's action control.
 * Variants: primary (petrol-teal fill), secondary (paper + border),
 * ghost (text only), danger (brick). Sizes: sm | md | lg.
 */
export function Button({
  children,
  variant = "primary",
  size = "md",
  iconLeft = null,
  iconRight = null,
  disabled = false,
  as = "button",
  href,
  onClick,
  style = {},
  ...rest
}) {
  const sizes = {
    sm: { padding: "0.4rem 0.8rem", fontSize: "var(--text-sm)", gap: "0.4rem" },
    md: { padding: "0.6rem 1.15rem", fontSize: "var(--text-body)", gap: "0.5rem" },
    lg: { padding: "0.8rem 1.5rem", fontSize: "var(--text-lead)", gap: "0.6rem" },
  };

  const variants = {
    primary: {
      background: "var(--primary)",
      color: "var(--on-primary)",
      border: "1px solid var(--primary)",
    },
    secondary: {
      background: "var(--surface)",
      color: "var(--ink)",
      border: "1px solid var(--line-2)",
    },
    ghost: {
      background: "transparent",
      color: "var(--primary)",
      border: "1px solid transparent",
    },
    danger: {
      background: "var(--bad)",
      color: "#fff",
      border: "1px solid var(--bad)",
    },
  };

  const base = {
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

  const hoverBg = {
    primary: "var(--primary-strong)",
    secondary: "var(--paper-2)",
    ghost: "var(--primary-tint)",
    danger: "#9a3d33",
  }[variant];

  const handlers = disabled
    ? {}
    : {
        onMouseEnter: (e) => {
          e.currentTarget.style.background = hoverBg;
          if (variant === "secondary") e.currentTarget.style.borderColor = "var(--line-3)";
        },
        onMouseLeave: (e) => {
          e.currentTarget.style.background = variants[variant].background;
          if (variant === "secondary") e.currentTarget.style.borderColor = "var(--line-2)";
        },
        onMouseDown: (e) => { e.currentTarget.style.transform = "translateY(1px)"; },
        onMouseUp: (e) => { e.currentTarget.style.transform = "translateY(0)"; },
      };

  const content = (
    <>
      {iconLeft}
      {children}
      {iconRight}
    </>
  );

  const Tag = href ? "a" : as;
  return (
    <Tag style={base} href={href} onClick={disabled ? undefined : onClick} disabled={as === "button" ? disabled : undefined} {...handlers} {...rest}>
      {content}
    </Tag>
  );
}
