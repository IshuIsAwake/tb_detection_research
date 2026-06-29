Segmented control with mono labels and a teal active pill. Use for confidence-threshold panels, view toggles, or any small mutually-exclusive switch.

```jsx
<Tabs
  items={[{ id: "0.25", label: "conf 0.25" }, { id: "0.5", label: "conf 0.5" }]}
  defaultValue="0.25"
  onChange={(id) => setConf(id)}
/>
```

- Self-managed from `defaultValue`, or controlled via `value` + `onChange`.
- Keep labels short; they're set in monospace.
