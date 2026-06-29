# Ideas / future directions

Parked ideas — noted, not built. Most are for a later custom-architecture or
multi-model stage, not the current YOLO detection baseline. Rough by design;
each needs a real design pass before it becomes an experiment.

## Pipeline topology (how to stage detectors and classifiers)

The long-term plan is two-stage (image classifier → TB detector). But the staging
is itself a design space worth measuring, not a fixed decision. Each stage can be
a detector or a classifier, and order matters. Things to test:

- **Classifier → detector.** Stage 1 classifies the whole image (healthy / sick /
  TB) and only forwards likely-TB images. Stage 2 detector localises the lesions
  and labels them (active vs obsolete, or active vs obsolete vs no-TB). This is
  the current intended pipeline — the classifier owns specificity so the detector
  can stay positives-only and sensitive.
- **Detector → classifier.** Stage 1 detector finds candidate lesions; stage 2
  classifier judges each crop (real lesion? active vs obsolete?). Lets a strong
  crop classifier clean up the detector's false positives.
- **Classifier → detector → classifier (3-stage).** Stage 1 classifier drops
  healthy + non-TB sick. Stage 2 detector finds lesions on the survivors. Stage 3
  classifier does active vs obsolete on each lesion crop. Splits the easy
  image-level call, the localisation, and the fine-grained lesion call into three
  specialised models.
- **Baselines / ablations to compare against:** detector-only (today's setup),
  detector → detector (a second detector re-scoring crops), classifier → classifier
  (no localisation at all — image-level only).

Open question for all of these: shared CXR-pretrained backbone across stages vs
separate models, and where the error compounds vs cancels.

## Augmentation

- **Anatomy-preserving mosaic.** A quadrant-constrained CutMix that only places a
  region where it anatomically belongs (a top-left lung stays top-left), so the
  composite is still a plausible chest. Plain YOLO mosaic just juxtaposes four
  partial chests with no anatomy constraint.
- **Custom lesion copy-paste.** Crop a labelled lesion box and paste it into a
  plausible lung field on another image, to multiply rare positives. Ultralytics'
  built-in `copy_paste` needs segmentation masks, so it's a no-op on our bbox-only
  labels — this would be custom. Must respect anatomy (paste into lung, not
  mediastinum or off-body). Overlaps with anatomy-preserving mosaic.
- **Lesion-size-conditional blur.** Blur only images whose lesions are large
  enough to survive it — acquisition-sharpness robustness without erasing the
  small lesions we're trying to find.

## Preprocessing / enhancement

- **Lung segmentation masks.** Segment the lung field (e.g. train on Montgomery /
  Shenzhen / JSRT lung masks) and ignore everything outside it — bones, spine,
  mediastinum, image borders. Stops the detector wasting capacity on non-lung
  structure and should cut spurious boxes. Likely a separate segmentation model
  feeding the detector (multi-model), so it needs the mask data first. Powerful if
  we have it; needs more thought.
- **Texture-based lesion enhancement.** TB lesions show up as heterogeneous
  patches — bright opacities mixed into darker lung fields. Compute a local texture
  map over the masked lung (rolling std-dev, local Shannon entropy,
  difference-of-Gaussians) and amplify the high-texture regions before detection.
  CLAHE is the standard, cruder cousin of this. Rough idea, unstructured — would
  need a real design pass, and could be made learnable (e.g. kornia) so the
  enhancement trains end-to-end with the detector rather than being fixed.
