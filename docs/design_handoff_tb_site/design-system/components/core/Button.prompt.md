Action control — primary actions, nav CTAs, and links across the research site. Use `primary` for the single main action, `secondary`/`ghost` for everything else, `danger` only for destructive actions.

```jsx
<Button variant="primary" iconRight={<i data-lucide="arrow-up-right" />}>
  View ablation studies
</Button>
<Button variant="secondary">Dataset research</Button>
<Button variant="ghost" size="sm">References</Button>
```

- **variant**: `primary` (petrol-teal fill) · `secondary` (paper + hairline border) · `ghost` (text only, teal) · `danger` (brick).
- **size**: `sm` · `md` · `lg`.
- Pass `href` to render an `<a>`; pass Lucide icon elements via `iconLeft` / `iconRight`.
- Labels are sentence case — never ALL CAPS.
