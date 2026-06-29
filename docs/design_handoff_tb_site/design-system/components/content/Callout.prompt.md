Tinted aside that carries the project's intellectual-honesty motif. Four kinds, each with its own color, icon and default label.

```jsx
<Callout kind="decision">Specificity belongs in the classifier; keep the detector positives-only.</Callout>
<Callout kind="landmine">COCO JSON is the box coord source, NOT the XML.</Callout>
<Callout kind="retraction"><s>512@16 clearly beats 1024@8</s> — anchored on one lucky seed.</Callout>
<Callout kind="note" title="Citation note">YOLOv8 has no peer-reviewed paper.</Callout>
```

- **kind**: `note` (info blue) · `decision` (good green) · `landmine` (warn clay) · `retraction` (bad brick).
- For retractions, wrap the wrong read in `<s>…</s>` and follow with the correction — never delete it.
- `title` overrides the default kind label.
