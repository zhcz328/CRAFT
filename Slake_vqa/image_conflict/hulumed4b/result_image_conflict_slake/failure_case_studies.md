# SLAKE Image-Conflict Failure Case Study

## Scope

This note focuses on one representative sample from:

- `ablate/ablate_selected_heads_all_val.json`

Selection rule:

- the no-conflict branch is correct before intervention
- the conflict branch is already failing before intervention
- the conflict branch still fails after intervention

This is the cleanest setup for answering the question:

> Which samples still fail after intervention, and why?

## Repository-Level Pattern

For the selected-head method, the dominant persistent-failure pattern in this directory is:

- `base_ctx_pred = unknown`
- `ab_ctx_pred = unknown`

So the main issue is not that the intervention creates a new wrong answer. The bigger issue is that the model stays trapped in an **abstention state**: conflict suppression does not rebuild enough confidence in the gold answer.

## Chosen Case

- Sample ID: `train:4483`
- QID: `4483`
- Question: `Where is the bladder existing in this image?`
- Category: `position`
- Question type: `target_location`
- Conflict target labels: `["Bladder"]`
- Gold answer: `Center`
- Image: `/root/autodl-tmp/data/SLAKE/imgs/xmlab612/source.jpg`
- Detection metadata: `/root/autodl-tmp/data/SLAKE/imgs/xmlab612/detection.json`
- Mask: `/root/autodl-tmp/data/SLAKE/imgs/xmlab612/mask.png`

Detection metadata for this image explicitly contains:

- `Bladder`
- `Left Femoral Head`
- `Right Femoral Head`
- `Rectum`
- `Small Bowel`

So this is not a case where the target object is absent from the auxiliary localization signal. The target is present and localized.

## What Happens Before And After Intervention

### No-conflict branch

- Before intervention: `base_nc_pred = gold`
- After intervention: `ab_nc_pred = unknown`

Logit summary:

- before: `gold_lp = -10.8125`, `unknown_lp = -12.5625`
- after: `gold_lp = -11.4375`, `unknown_lp = -10.0000`

This means the model originally answers the clean sample correctly, but intervention destroys that advantage and pushes the no-conflict branch into `unknown`.

### Conflict branch

- Before intervention: `base_ctx_pred = unknown`
- After intervention: `ab_ctx_pred = unknown`

Logit summary:

- before: `gold_lp = -12.1875`, `unknown_lp = -10.3125`
- after: `gold_lp = -13.3750`, `unknown_lp = -9.1875`

Equivalent gold-vs-unknown gap:

- before: `-1.8750`
- after: `-4.1875`

So the intervention does not merely fail to rescue the sample. It makes the abstention regime stronger.

## Why This Case Still Fails

### 1. The method suppresses confidence but does not reconstruct localization reasoning

This question is not a yes/no presence check. The model must convert visual evidence into a **spatial answer**: `Center`.

The image and detection metadata both contain enough evidence that the bladder is the central structure. But selected-head ablation does not inject a stronger center-location representation. It only perturbs the existing decision path.

Net effect:

- the wrong option is already irrelevant
- the real competition is `gold` vs `unknown`
- intervention fails because it cannot lift `Center` above `unknown`

### 2. The intervention causes collateral damage on the clean branch

This sample has:

- `nc_gold_margin_damage = 3.1875`

That is one of the larger damages among the persistent SLAKE failures. The intervention harms the no-conflict branch enough that a previously correct clean prediction also becomes `unknown`.

This is important because it shows the failure is not just “the conflict sample was too hard.” The selected heads here are entangled with **general answer-supporting evidence**, not only with the harmful conflict pathway.

### 3. The failure mode is abstention, not contradiction

The model never flips to a confident wrong anatomical location. Instead, it retreats to `unknown`.

That tells us the current method is better described as:

- a **suppression mechanism**

than as:

- a **recovery mechanism**

It can damp some harmful activations, but in this case it cannot rebuild the missing grounded answer.

## Takeaway

This failure case suggests a concrete limitation of the method on SLAKE image-conflict:

- for **target-location** questions, removing selected heads is often insufficient because the missing capability is not only conflict suppression, but also **fine-grained grounded localization**
- when the target answer competes mainly with `unknown`, intervention may even strengthen abstention instead of restoring the gold answer
- the same selected-head intervention can damage the clean branch, showing that the affected heads also carry useful visual-answer evidence

In short, this sample fails because the method can suppress a pathway, but it cannot reconstruct the spatial grounding needed to answer `Center`.
