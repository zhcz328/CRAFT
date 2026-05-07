# VQA-RAD Failure Case Study

## Scope

This note analyzes one representative failure from:

- `ablate_selected_heads_all_hulumed4b_val.json`

Selection rule:

- the no-conflict branch is correct before intervention
- the conflict branch predicts `wrong` before intervention
- the conflict branch is still `wrong` after intervention

This isolates the most informative type of failure:

> the method moves the logits in the right direction, but still cannot fully recover the gold answer.

## Repository-Level Pattern

In this directory, persistent failures are different from the SLAKE image-conflict setting.

For the selected-head method:

- persistent failures are almost entirely `wrong -> wrong`
- there is little evidence of collapse to `unknown`

So the main limitation here is not abstention. It is **insufficient semantic correction**: the intervention weakens the conflict-favored option, but not enough to reverse the final ranking.

## Chosen Case

- Row index: `330`
- Sample ID: `351`
- Image: `synpic38263.jpg`
- Image path: `/root/logit_lens/VQA_RAD/Hulu-med/data/VQA_RAD_Image_Folder/synpic38263.jpg`
- Category: `other`
- Question: `what is the cause of this finding medical process or physical injury`
- Gold answer: `medical process`
- Conflict answer: `physical injury`

## What Happens Before And After Intervention

### No-conflict branch

- Before intervention: `base_nc_pred = gold`
- After intervention: `ab_nc_pred = gold`

Logit summary:

- before: `gold_lp = -8.6780`, `wrong_lp = -11.1309`
- after: `gold_lp = -8.6023`, `wrong_lp = -12.0645`

So the clean branch stays correct, and the intervention even enlarges the gold-over-wrong margin there.

### Conflict branch

- Before intervention: `base_ctx_pred = wrong`
- After intervention: `ab_ctx_pred = wrong`

Logit summary:

- before: `gold_lp = -13.2957`, `wrong_lp = -9.2539`
- after: `gold_lp = -9.8268`, `wrong_lp = -8.9355`

Gold-vs-wrong margin:

- before: `-4.0417`
- after: `-0.8913`

So the intervention clearly helps:

- the gold answer gains a lot
- the wrong answer loses relative advantage

But the final ranking still does not flip. The sample remains wrong.

## Why This Case Still Fails

### 1. This question requires etiology-level reasoning, not just conflict suppression

The prompt is not asking for a visible object name or a simple binary finding. It asks for the **cause** of the observed pattern:

- `medical process`
- vs `physical injury`

That is a higher-level semantic judgment. Even if selected-head ablation reduces spurious conflict-following, it does not add the domain knowledge needed to map the image pattern to the correct causal interpretation.

In other words, the intervention can weaken the bad answer without supplying the missing diagnostic abstraction.

### 2. The method improves the margin but cannot cross the decision boundary

This case is useful because it is not a total failure. The method moves the sample substantially:

- margin improves from `-4.0417` to `-0.8913`

That tells us the selected heads do participate in the harmful conflict pathway. But it also tells us those heads are only part of the story.

The remaining error must still be supported by:

- other layers or heads that were not ablated
- model priors that favor the conflict answer
- insufficient visual-semantic evidence for the gold label in the residual stream

### 3. The clean branch staying correct reveals a different failure mode than SLAKE

Unlike the SLAKE case, this sample does not collapse into `unknown`, and the no-conflict branch remains healthy.

So this is not an over-suppression failure. It is a **partial-repair failure**:

- the intervention is directionally correct
- but the corrective effect is too weak for a hard semantic distinction

## Takeaway

This VQA-RAD case shows the second major limitation of the method:

- when the target error depends on **high-level clinical interpretation**, ablating selected heads may reduce the wrong answer’s dominance but still fail to produce the correct answer
- success on simpler conflict settings does not guarantee success on semantically abstract questions such as cause attribution
- the remaining error indicates that harmful conflict behavior is distributed beyond the currently selected head set

In short, this sample fails because the intervention weakens `physical injury`, but cannot provide enough evidence for `medical process` to become the final answer.
