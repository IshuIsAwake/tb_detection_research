Small uppercase status pill. Carries result/status meaning through color + a word — used for experiment outcomes, dataset roles, and run state.

```jsx
<Badge tone="good">Finalist</Badge>
<Badge tone="bad">Ruled out</Badge>
<Badge tone="warn">Capped ~60%</Badge>
<Badge tone="primary" variant="solid">Latest run</Badge>
```

- **tone**: `good` · `warn` · `bad` · `info` · `primary` · `neutral`.
- **variant**: `soft` (tinted, default) · `solid` (filled).
- Keep labels short; the badge is uppercase by design.
