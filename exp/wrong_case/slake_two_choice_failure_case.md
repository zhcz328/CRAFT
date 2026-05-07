# SLAKE Failure Case Study: Explicit Two-Choice Question

## Scope

This note records one SLAKE image-conflict failure case where the answer space is an explicit two-way choice.  
The focus here is the relationship among:

- `gold`
- `wrong`
- `unknown`

and how that relationship changes:

- **before ablation**
- **after ablation**

Source result file:

- `/root/logit_lens/Slake_vqa/image_conflict/hulumed4b/result_image_conflict_slake/ablate/ablate_selected_heads_all_val.json`

## Chosen Case

- Sample ID: `test:12086`
- QID: `12086`
- Question: `Which is bigger in this image, kidney or liver?`
- Category: `size`
- Question type: `size_compare`
- Gold answer: `Liver`
- Wrong answer: `Kidney`
- Image: `/root/autodl-tmp/data/SLAKE/imgs/xmlab208/source.jpg`
- Detection metadata: `/root/autodl-tmp/data/SLAKE/imgs/xmlab208/detection.json`
- Mask: `/root/autodl-tmp/data/SLAKE/imgs/xmlab208/mask.png`

Detection metadata for this sample contains only:

- `Liver: [76.0, 128.0, 137.0, 178.0]`

So the labeled cue is already biased toward the gold object rather than the wrong object.

## What The Model Looks Like Before Ablation

### No-conflict branch

Before ablation, the clean branch is correct:

- `base_nc_pred = gold`

Score relation:

- `gold_lp = -9.6875`
- `wrong_lp = -9.7500`
- `unknown_lp = -10.6875`

Ranking before ablation on the clean branch:

- `gold > wrong > unknown`

This means the model can answer the question correctly when there is no conflict signal.

### Conflict branch

Before ablation, the conflict branch is wrong:

- `base_ctx_pred = wrong`

Score relation:

- `gold_lp = -10.0625`
- `wrong_lp = -8.5000`
- `unknown_lp = -10.1875`

Ranking before ablation on the conflict branch:

- `wrong > gold > unknown`

This ordering is important.  
The failure **before ablation** is not an `unknown` failure. The model is making a **confident wrong choice**:

- `Kidney` is ranked highest
- `Liver` is second
- `unknown` is lowest

So before ablation, the problem is:

- the model is pulled toward the wrong answer
- not that it is uncertain

## What The Model Looks Like After Ablation

### No-conflict branch

After ablation, the clean branch remains correct:

- `ab_nc_pred = gold`

Score relation:

- `gold_lp = -9.5000`
- `wrong_lp = -9.7500`
- `unknown_lp = -9.5000`

Ranking after ablation on the clean branch:

- `gold ≈ unknown > wrong`

So ablation does not destroy the clean branch.  
The no-conflict prediction is still recoverable, and the damage is very small:

- `nc_gold_margin_damage = 0.0625`

### Conflict branch

After ablation, the conflict branch is still wrong:

- `ab_ctx_pred = wrong`

Score relation:

- `gold_lp = -10.1250`
- `wrong_lp = -8.3750`
- `unknown_lp = -9.1875`

Ranking after ablation on the conflict branch:

- `wrong > unknown > gold`

This is the key point of the case.

After ablation, the relation among the three candidates changes from:

- **before**: `wrong > gold > unknown`

to:

- **after**: `wrong > unknown > gold`

So the ablation does two things:

1. it does **not** remove `wrong` from the top position
2. it pushes `unknown` above `gold`

That means the intervention not only fails to promote the gold answer; it actually makes the gold answer less competitive relative to `unknown`.

## Before vs After: Three-Way Analysis

### `wrong` vs `gold`

Before ablation:

- `gold_wrong_margin = -1.5625`

After ablation:

- `gold_wrong_margin = -1.7500`

So relative to the wrong answer, the gold answer becomes slightly worse after ablation.

### `gold` vs `unknown`

Before ablation:

- `gold - unknown = 0.1250`

After ablation:

- `gold - unknown = -0.9375`

This is the most important shift.

Before ablation:

- `gold` is still above `unknown`

After ablation:

- `unknown` rises above `gold`

So ablation weakens the gold answer enough that it falls below `unknown`.

### `wrong` vs `unknown`

Before ablation:

- `wrong - unknown = 1.6875`

After ablation:

- `wrong - unknown = 0.8125`

This part actually improves:

- the dominance of `wrong` over `unknown` becomes smaller

This is consistent with:

- `hallucination_relief = 0.8125`

So the method is suppressing some harmful conflict preference.  
But that suppression is not enough to make `gold` win. Instead, it mostly changes the lower part of the ranking:

- from `gold > unknown`
- to `unknown > gold`

while `wrong` still stays on top.

## Why This Case Still Fails

### 1. Before ablation, the failure is a wrong-answer failure, not an uncertainty failure

The pre-ablation ranking is:

- `wrong > gold > unknown`

So the model is already leaning toward a specific incorrect comparative judgment, not merely abstaining.

### 2. After ablation, the model does not repair the top choice

The post-ablation ranking is:

- `wrong > unknown > gold`

So the method fails in two senses:

- it does not overturn the wrong top answer
- it pushes the gold answer below `unknown`

This means the intervention weakens useful gold-supporting evidence more than it weakens the final wrong preference.

### 3. The method changes the three-way geometry in the wrong direction for `gold`

The effect of ablation is not simply “too weak.”  
It changes the geometry of the three candidates like this:

- `wrong` remains first
- `unknown` moves up
- `gold` moves down to last

So this case is exactly the kind of example where analyzing only `wrong -> wrong` is not enough.  
The more informative interpretation is:

- the method partially suppresses conflict
- but the residual competition shifts from `wrong vs gold`
- into `wrong vs unknown`, with `gold` falling behind both

## Takeaway

For this SLAKE two-choice failure, the right way to understand the error is through the three-way relation among `wrong`, `gold`, and `unknown`.

Before ablation:

- `wrong > gold > unknown`

After ablation:

- `wrong > unknown > gold`

So the method does not rescue the sample.  
Instead, it preserves the wrong answer at the top while making the gold answer even less competitive by letting `unknown` overtake it.

This suggests that in at least some explicit two-choice image-conflict cases, the current ablation method is not truly reconstructing the correct answer pathway. It is only weakening part of the competition, and the final result is still failure.
