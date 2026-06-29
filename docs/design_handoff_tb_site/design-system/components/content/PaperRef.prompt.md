Literature reading-list row — the backbone of the References surface. Shows title (serif), authors/venue, a "why it matters here" note, a mono arXiv link, and a personal read-tracker checkbox.

```jsx
<PaperRef
  important
  title="ResNet — Deep Residual Learning for Image Recognition"
  meta="He et al. · CVPR 2016"
  link="https://arxiv.org/abs/1512.03385"
  note="The skip-connection paper. Backbone candidate for both stages."
/>
```

- **important**: teal left-accent + "Must read" tag for non-negotiable citations.
- Checkbox is self-managed; pass `read` + `onToggleRead` to control it (e.g. persist progress).
- `linkLabel` overrides the shown link text.
