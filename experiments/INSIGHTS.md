# Object-Centric HIRO — Insights Learned Through Failure

## 0. One-paragraph summary
We built a two-level HIRO agent on ManiSkill PickCube and spent the campaign systematically
killing failure modes. The headline: a **from-scratch, object-centric HIRO solves PickCube 100%**,
and the recipe that works is **object-only (cube-centric) subgoal + HIRO's off-policy correction
turned OFF + a hybrid worker reward (w=0.5) + gamma=0.8**. Every ingredient was *earned* by watching
the alternative fail. The honest limit: on this *single-stage* task the learned **manager is not
load-bearing** — the solve is driven by the environment reward through a capable worker, not by the
manager's subgoals (proven cleanly: the same config solves at w=0.5 and collapses when the env
reward is annealed to w=0).

## 1. Setup
- ManiSkill 3 **PickCube-v1**, Panda, 8-D `pd_joint_delta_pos`, 42-D state obs, 50-step episodes.
  Reward (normalized = raw/5) = reach + grasp + place·is_grasped + static·is_obj_placed; success -> 5.
- Reference: flat SAC solves PickCube in ~200k steps at gamma=0.8.
- HIRO: manager emits a subgoal every c=10 steps; worker = SAC on [obs, subgoal]; hybrid worker
  reward (1-w)*intrinsic + w*env. gamma_low = gamma_high = 0.8. 131 unit tests.
- "Object-centric" = the subgoal is the **cube's 3-D position** (grasp-gated), not robot state.

## 2. The failure -> insight chain (the core)

**F1. Manager Q diverged to 1e8.** gamma_high=0.99 + always-bootstrapping blew up the manager's value.
  -> *Lesson:* bound the manager horizon (gamma_high=0.8) + grad clip. A high-level value over long
  horizons is unstable; keep it short and clipped.

**F2. Manager temperature (alpha) exploded 1 -> 81.** subgoal_scale=0.15 shifts a tanh-squashed
  policy's log-prob, breaking the standard target-entropy = -dim heuristic.
  -> *Lesson:* scale-correct the target entropy: `te = -dim + sum(log(action_scale))`. Auto-temperature
  SAC silently breaks if the action scale != 1 and you don't correct it.

**F3. gamma_low=0.95 was a silent killer.** The worker never learned to *place*; only gamma_low=0.8
  (the ManiSkill default) worked.
  -> *Lesson:* the discount must match the task horizon; a slightly-too-high gamma can silently
  prevent learning with no error, just a flat curve.

**F4. The universal 0.4 plateau.** Every run (ours + classmates' other encodings) plateaued at ~0.4
  normalized reward. Decoded: 0.4 = (reach 1 + grasp 1)/5 with the transport term ~0. Place & static
  rewards are **gated on is_grasped**, so fumbling the grasp during transport loses everything (a cliff).
  -> *Lesson:* the plateau is a **reward-landscape artifact**, not a representation problem. A gated
  transport reward punishes the exploration needed to cross it.

**F5. H1 (manager horizon) rejected.** Hypothesis: the manager's discount crushes the success signal.
  Reality: flat SAC solves at gamma=0.8 with a *shorter* effective horizon than the manager; and
  raising gamma_high to 0.95 didn't help (and inflated Q).
  -> *Lesson:* it was NOT a horizon/discount problem. (Killing a plausible hypothesis with a control
  is as valuable as confirming one.)

**F6. The hover-hack (the biggest fix).** The standard HIRO subgoal (tcp + cube positions) has an
  always-on TCP/"hand" term. The worker learned to satisfy it by **hovering the gripper near the
  commanded point without ever grasping** -> train grasp_rate stuck at **2-3%**, and
  `mean_intrinsic_reward -> 0` *while* grasp stayed ~2-3% (it "achieved" the subgoal by hovering).
  -> *Fix:* make the subgoal the **object only** (cube position, grasp-gated) -> no hand term to farm,
  so the only way to earn reward is to actually grasp and move the cube -> grasp **2% -> 53%**,
  place_dist 0.20 -> 0.115, reward -> 0.47.
  -> *Lesson:* a subgoal that can be satisfied *without manipulating the object* creates a farmable
  local optimum. Object-centric subgoals close that loophole.

**F7. Pure-intrinsic worker (w=0) can't bootstrap grasping.** With the env reward removed, the
  cube-centric intrinsic is **identically zero pre-grasp** (no hand term; cube term gated off; grasp
  bonus only fires *after* a grasp) -> no gradient to learn grasping. Even the **oracle** manager
  fails at w=0. A potential-based reach-to-cube bootstrap helped but was too weak (the one-time grasp
  bonus can't compete with the env reward's persistent grasp signal).
  -> *Lesson:* "bootstrap-hackability tension" -- the very property that makes the subgoal
  non-hackable (no hand term) also removes the pre-grasp learning signal. The worker needs the env
  reward (hybrid w>0) to bootstrap manipulation. Pure HIRO (w=0) is not viable here.

**F8. HIRO's off-policy correction is HARMFUL (counterintuitive).** HIRO's signature mechanism
  relabels the manager's subgoal toward "what the worker actually did." With a partly-freelancing
  worker, this taught the manager to **mimic the worker instead of lead it** -- the manager's subgoals
  pointed *away* from the goal (`sg_align = -0.38` with the correction ON). Turning it **OFF** is what
  took the cube-centric system from reward 0.47 / 0% success to a **100% solve**.
  -> *Lesson:* the off-policy correction assumes the worker follows the manager; when the worker also
  optimizes a task reward, the relabeling becomes a harmful "predict-the-worker" feedback loop.

**F9. Manager place-PBRS and more manager training (UTD) didn't help.** A policy-invariant place
  potential raised the manager's *intent* (sg_align) at w=0.5 but produced no transport; 4x manager
  gradient steps left it stuck.
  -> *Lesson:* the manager's problem was not under-training or reward shape -- it was (a) the harmful
  off-policy correction and (b) a deeper no-gradient issue (below).

**F10. The learned manager is NOT load-bearing on PickCube (the honest limit).** The decisive test:
  anneal the env weight w 0.5 -> 0. With the correction ON (Exp 1) *and* OFF (E2), the manager faded
  -- as w -> 0, grasp collapsed (1.0 -> ~0.3), reward collapsed, sg_align never sustained >= +0.2, and
  it stopped solving. Meanwhile the **oracle** (perfect manager) solves 100%, and M2 (same config,
  *constant* w=0.5) solves -- but M2 collapses under the w->0 anneal.
  -> *Lesson:* M2's solve is **env-reward-driven, not manager-driven**; the worker is fully capable
  but the *learned* manager is a partial/non-essential contributor. Root cause: at w=0.5 the worker
  freelances on the env reward, so the manager's subgoals barely affect the outcome -> the manager
  gets almost no gradient to improve. This is expected for a **single-stage** task; a manager only
  earns its keep when the task needs **temporal decomposition** (e.g., StackCube).

## 3. The recipe that worked (M2)
`hiro_M2_noOPC_cubeCentric_w50_seed1_300k`: **cube-centric (object-only) subgoal + off-policy
correction OFF + hybrid worker reward w=0.5 + gamma=0.8**, learned manager + learned worker, trained
**from scratch** (no oracle, no demos, no pretraining).
- Result: success 0.875 @250k -> **1.000 @275k & 300k**, reward 0.77, place_dist 0.066.
- First fully-LEARNED (no privileged info) solve of the campaign.

## 4. Consolidated lessons (the takeaways)
1. **Object-centric subgoals beat generic HIRO subgoals** -- not as a representation nicety, but
   because they remove a *farmable loophole* (the hover-hack). Pure-object subgoal > effector+object.
2. **HIRO's off-policy correction can hurt** when the worker isn't strictly subgoal-following. Drop it.
3. **You need a hybrid (task-reward) worker signal to bootstrap manipulation**; pure intrinsic fails.
   The "purity" you cannot have is on the reward; the purity that helps is on the subgoal.
4. **On a single-stage task the hierarchy adds little** -- the manager is not load-bearing; the env
   reward + a capable worker do the work. Test the manager's value on a multi-stage task.
5. **Reward is a trap metric.** A rising reward can hide a dead manager (env-driven). The right metric
   for "is the manager doing anything?" is **sg_align** (does the worker follow the manager's subgoal?).

## 5. Honest limits & open questions
- M2 is a **single seed** (reproduce on 2-3 seeds for a robust claim).
- The solve uses a **hybrid reward (w=0.5)** -- not pure HIRO.
- The **off-policy correction is disabled** -- a deviation from textbook HIRO (and itself a finding).
- The **manager is not load-bearing** on PickCube; whether a load-bearing learned manager is
  achievable on a multi-stage task (StackCube) is open.
- **Transfer** (the thesis claim) is untested; runnable on the M2 source but weaker while the manager
  isn't load-bearing.
- A **fully object-centric manager** (manager conditioned only on object/goal state, not robot joints)
  is untried -- the thesis-ideal, embodiment-invariant version; main payoff is transfer.

## 6. Methodology that made it work
- **Diagnostic ladder:** change exactly one variable from a known-good reference to isolate each cause.
- **Upper-bound controls:** an oracle (perfect manager) separated "is the worker capable?" (yes) from
  "is the learned manager good?" (the gap).
- **Quantitative reward decomposition:** decoding the 0.4 plateau (= reach+grasp) grounded the whole
  diagnosis in the reward function, not vibes.
- **Pick the right gating metric:** judging by reward repeatedly misled us; sg_align was the metric
  that actually tracked the manager's contribution.
- **Kill runs early / log everything:** flat-lining configs were stopped; every result logged.

## 7. Status & next steps
Validated: object-centric subgoal fixes the hover-hack; off-policy correction is harmful;
from-scratch object-centric HIRO solves PickCube 100% (M2). Not yet: load-bearing learned manager,
multi-seed reproduction, transfer. Natural next steps: (a) 2-3 seed reproduction of M2, (b) StackCube
(multi-stage -> where the manager should matter), (c) transfer on the M2 source.
