"""
inventory.py — Step 0 data sanity check, shared by every experiment.

Verifies the on-disk TBX11K data against data_research_report.md (folder counts,
799 positives, box histogram, class map) and HALTS on any contradiction rather
than guessing or silently "fixing". This enforces the repo's "when docs and code
disagree, surface it" rule for the data.
"""

from __future__ import annotations

from yolo_common import convert, settings as S

# Expected values from data_research_report.md / README — halt if reality differs.
# `extra` (External: DA/DB/Montgomery/Shenzhen) is out of scope and nested in
# subdirs, so it is reported for info only, never hard-checked.
EXPECT_FOLDERS = {"tb": 800, "health": 3800, "sick": 3800, "test": 3302}
EXPECT_POSITIVES = 799
EXPECT_BOXES = {"ActiveTuberculosis": 972, "ObsoletePulmonaryTuberculosis": 239}
EXPECT_TOTAL_BOXES = 1211


def inventory_or_halt() -> dict:
    print("══ Step 0 — inventory (halt on contradiction) ══")
    problems = []

    for k, exp in EXPECT_FOLDERS.items():
        got = len(S.list_images(S.IMGS_ROOT / k))
        flag = "OK" if got == exp else "MISMATCH"
        if got != exp:
            problems.append(f"folder {k}: expected {exp}, got {got}")
        print(f"  imgs/{k:<7} {got:>5}  ({flag})")
    extra_n = sum(1 for _ in (S.IMGS_ROOT / "extra").rglob("*")
                  if _.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"  imgs/extra   {extra_n:>5}  (info only — External, out of scope)")

    rep = convert.build_or_load()
    n_pos = rep["n_positive_images"]
    if n_pos != EXPECT_POSITIVES:
        problems.append(f"positives: expected {EXPECT_POSITIVES}, got {n_pos}")
    print(f"  positives (non-empty labels): {n_pos}  "
          f"({'OK' if n_pos == EXPECT_POSITIVES else 'MISMATCH'})")

    for name, exp in EXPECT_BOXES.items():
        got = rep["box_counts"].get(name, 0)
        if got != exp:
            problems.append(f"box count {name}: expected {exp}, got {got}")
        print(f"  boxes {name:<30} {got:>5}  "
              f"({'OK' if got == exp else 'MISMATCH'})")
    if rep["total_boxes"] != EXPECT_TOTAL_BOXES:
        problems.append(f"total boxes: expected {EXPECT_TOTAL_BOXES}, got {rep['total_boxes']}")
    print(f"  total boxes: {rep['total_boxes']}  "
          f"({'OK' if rep['total_boxes'] == EXPECT_TOTAL_BOXES else 'MISMATCH'})")

    if rep["class_map"] != {"0": "ActiveTuberculosis", "1": "ObsoletePulmonaryTuberculosis"}:
        problems.append(f"class map flipped/wrong: {rep['class_map']}")
    print(f"  class map: {rep['class_map']}")
    print(f"  signatures: {rep['class_signature_counts']}")

    if problems:
        print("\n!! CONTRADICTION with data_research_report.md — HALTING:")
        for p in problems:
            print(f"   - {p}")
        raise SystemExit(1)
    print("  → no contradiction.\n")
    return rep
