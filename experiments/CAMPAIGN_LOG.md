# HIRO 0.4-Plateau Campaign — Log (append-only)

## Reference / setup
- Task: ManiSkill3 **PickCube-v1**, Panda, `pd_joint_delta_pos`. Episode horizon = **50 steps**.
  Dense reward = reach + grasp + place·is_grasped + static·is_obj_placed; success sets reward=5; normalized = raw/5.
  Milestones (normalized): reach 0.2 · reach+grasp ≈ **0.4** · cube-at-goal 0.6 · placed+static 0.8 · success 1.0.
- Known-good reference: flat ManiSkill SAC solves PickCube in ~200k steps.
- HIRO: manager (c=10, 6-D object-centric subgoal, scale 0.15, γ_high=0.8) + worker (γ_low=0.8, hybrid reward w=0.5, PBRS grasp potential). Plateaus at ~0.4 reward / ~0% success.

---

## Phase 0 — instrumentation + dispositive cheap test

### 0b. flat SAC at γ=0.8 — RESOLVED WITHOUT A RUN (source/config evidence) [2026-05-30]
- ManiSkill `examples/baselines/sac/sac.py` Args default is `gamma: float = 0.8` (confirmed from source).
  `modal_train_sac.py` passes **no** `--gamma`, so the solving baseline ran at γ=0.8.
- => **Prediction B realized: flat-γ0.8 does NOT plateau.** **H1 (the 0.4 plateau is a γ=0.8 discount/horizon artifact) is REJECTED by existing evidence.** No 3-seed Phase 0b run needed (compute saved).

### Mechanism check (`hiro_agent.update_high`)
- γ_high is applied **per manager transition** (s0→s_c spans c=10 env-steps); manager reward = `reward_sum` (undiscounted sum of shaped env reward over the segment).
- Manager effective horizon = 1/(1−0.8) = 5 manager-steps = **50 env-steps = a full episode**.
  Flat SAC at γ=0.8 has a **5-env-step** horizon and still solves PickCube. The manager's horizon is ~10× longer and it fails → **horizon is not the bottleneck. H1 dead twice over.**

### Reframed mechanism (replaces H1)
- The manager's objective is maximized by **grasp-and-hold**: `reward_sum` banks ~2/step ≈ 20/segment for holding; commanding transport subgoals risks worker drops, which collapse `reward_sum` (place & grasp gated on is_grasped → cliff). Coarse manager exploration (5 decisions/episode, 6-D) gets punished whenever it probes transport.
- = **H3 (cliff basin) realized at the manager level**, plausibly + **H2** (manager sees ~30k segment-transitions over 300k env-steps vs ~200k flat SAC needs).
- **Lever = manager reward landscape + exploration, not discount.** Reprioritize: Phase 2 (manager place-PBRS) and Phase 5 (cube-centric grasp-gated subgoal) first; Phase 1 (γ_high sweep) demoted to one confirmatory screen; Phase 3 (length/UTD) folded into promotions; Phase 4 (HER) contingent.

### 0a. instrumentation — PENDING (reward decomposition + manager-Q/alpha/entropy + subgoal-direction logging)

---

## Decisions / budget — RESOLVED [2026-05-31]
- Stopped 2 wasted +1.5M resumes. User approved reframed plan + screening-first budget (screen=1, decisive={1,2,3}).

---

## Phase 2 results — place-PBRS beta screens (seed1, 200k) [2026-05-31]
Reference config: w=0.5, gamma_low=0.8, gamma_high=0.8, no-partial-reset; vary ONLY place_pbrs_beta.

| beta | success | reward(200k) | ever_grasped | place_dist | subgoal_goal_align | train grasp_rate |
|------|---------|--------------|--------------|------------|--------------------|------------------|
| 0 (control) | 0 | 0.171 | 0.50 | 0.23 flat | ~0.1 noisy (±) | 0.02 |
| 0.5  | 0 | 0.232 | 0.69 | 0.22 flat | **~0.5 (UP)** | 0.03 |
| 1.0  | 0 | 0.153@160k | 0.31 | 0.28 | rising->0.52 | - |
| 2.0  | 0 | 0.251 | 0.81 | 0.23 flat | ~0.05 noisy | 0.03 |

VERDICT: place-PBRS does NOT cross the ridge at any beta (success=0, place_dist flat). BUT beta=0.5 cleanly raised subgoal_goal_align (~0.1 -> ~0.5) at the same subgoal magnitude (~0.20): the PBRS gradient successfully makes the MANAGER command transport. Manager Q bounded (8-12), no instability.
**KEY DIAGNOSTIC: data/grasp_rate during TRAINING is ~2-3% in all runs.** The worker barely holds the cube, so transport reward (manager or worker) has almost no grasped time to act on. **Bottleneck = WORKER grasp-stability, not manager reward shaping.** Manager subgoal magnitude ~0.20 (not collapsed to "stay").

## Phase 1 result — gamma_high=0.95 screen [2026-05-31]
gamma_high 0.8->0.95 (beta=0): reward 0.18, place_dist flat 0.21, success 0; manager q_max peaked ~20 (vs ~13 at 0.8) then settled. More horizon did NOT help and inflated Q. => **H1 empirically CONFIRMED DEAD.**

## DECISION (post-Phase-2) [2026-05-31]
Bottleneck is worker grasp-stability (grasp_rate ~2-3%), not manager reward. place-PBRS beta=0.5 is a GOOD manager-side fix (raises transport intent) but is wasted on a grasp-incapable worker. So: give place-PBRS a grasp-CAPABLE worker.
LAUNCHED (no new code): pretrained worker (Test-H oracle-hybrid-w50 final_ckpt, eval ever_grasped~0.94) + place-PBRS, beta in {0.5, 1.0}, w=0.5, g08, no-PR, learning_starts=0, 200k seed1.
  - hiro_p2b_PRETRw_pPBRS_b05_g08_seed1_200k
  - hiro_p2b_PRETRw_pPBRS_b10_g08_seed1_200k
Hypothesis: competent grasping worker (high grasp_rate) + transport-intending manager (place-PBRS) = transport. FULL#2 (pretrained worker, beta=0) grasped 94% but never placed (no manager intent); this adds the intent. If THIS still fails, the manager's learned subgoals are too weak/noisy even for a good worker, and the lever becomes Phase 5 (cube-centric subgoal) / worker-side grasp robustness.

## DEEP EXTRACTION from saved dumps (free, no compute) [2026-05-31]
Mined worker exploration + reward-decomposition tags across all screens:
1. Worker low/alpha 0.99->0 by ~80k; low/entropy +5.4 -> -8.0 (= te_low target). NORMAL SAC convergence to the entropy target, NOT a pathological collapse (flat SAC does the same and solves). => entropy floor is likely the WRONG lever.
2. **mean_intrinsic_reward -> ~0 while grasp_rate stays 2-3%.** The worker drives the subgoal residual to ~0 WITHOUT grasping => it satisfies the always-on hand-term by HOVERING. Quantitative confirmation that the hover-hack (not exploration death) causes the low grasp rate. Phase 5 (cube-centric, no hand-term) removes exactly this escape.
3. place-PBRS confirmed POLICY-INVARIANT: mean_manager_reward beta=0.5 (->1.02) ~= beta=0 (->1.04). Reshaped the gradient (sg_align 0.1->0.5) without inflating return (Ng et al.). Clean mechanism, just starved by low grasp.
4. Pretrained worker low/alpha=0.001 flat (near-deterministic from load) yet competent; mean_manager_reward ->2.26 (~2x from-scratch's ~1.0). Same ~0 exploration, opposite grasp (95% vs 2-3%) => discriminator is converged-policy QUALITY (reward landscape), not exploration level.
CONCLUSION: from-scratch wall = hover-hackable reward. Phase 5 (cube-centric) is the principled fix; its running screens are the direct test (watch grasp_rate + mean_intrinsic_reward rise together).

## MILESTONE: FIRST FULLY-LEARNED SOLVE -- M2 (no off-policy correction) [2026-06-01]
M2 = cube-centric, w=0.5, learned manager + learned worker, **off-policy correction OFF**, 300k.
RESULT: success 0.875@250k -> **1.000@275k & 300k**, reward 0.773, place_dist 0.066, cube@goal 0.70. FIRST fully LEARNED (no-oracle, no privileged info) solve of the campaign (all prior learned configs plateaued ~0.47/0).
sg_align ROSE as it converged: +0.13(250k)->+0.235(275k)->+0.247(300k) -- the no-OPC manager gets MORE engaged (vs correction-ON runs which faded to ~0).
CAUSE = removing the off-policy correction. M1 (same + 4x manager UTD but correction ON) stuck at reward ~0.42, success 0, sg_align -0.28. => HIRO's subgoal-relabeling was ACTIVELY HARMING (teaching the manager to mimic the worker, not lead). Turning it off unlocked the solve. UTD irrelevant.
CALIBRATION: solves at w=0.5 with sg_align ~0.25 (<< oracle 0.83), so the manager is PARTIALLY steering; env reward still shares the load. Not yet a w=0 load-bearing claim.
NEXT (decisive + reshaped): (1) anneal w->0 WITH no-OPC -- the earlier "not load-bearing" verdict used correction-ON runs; now very promising. (2) TRANSFER on the LEARNED M2 source (no oracle needed -- much stronger thesis claim).

## ANNEAL VERDICT (Exp 1 to w->0) -- REFINED with 1a w=0 endpoint [2026-05-31]
CORRECTION to earlier "leans fundamental": 1a's w=0 endpoint = sg_align +0.282 (ABOVE +0.2), grasp 0.75, place_dist 0.172, reward 0.30, success 0. So the worker STILL FOLLOWS the learned manager at w=0 => manager is GENUINELY (weakly) LOAD-BEARING, not a passenger. But performance DEGRADES as w->0 (grasp 1.0->0.75, place_dist best 0.115@w~0.3 -> 0.172@w0, reward 0.45->0.30, never solves). 1b (+place-PBRS) WORSE (sg_align +0.125, grasp 0.50) -> place-PBRS hurt.
REFINED VERDICT: neither clean-fundamental nor clean-phase-separable. The learned hierarchy is REAL (manager followed at w=0) but the learned manager's SUBGOAL QUALITY is too low to run the task alone; peak is an intermediate sweet spot (~w=0.3). Combined with oracle (sg_align 0.83 -> 100% solve): worker fully capable, manager weakly load-bearing, manager subgoal QUALITY is the single remaining gap.
NEXT (well-motivated now): (1) manager-UTD screen (4x manager grad steps) to lift learned subgoal quality -- since the manager IS followed, better subgoals should translate to performance. (2) transfer experiment on oracle source (deliverable). All Exp-1 runs done; nothing running.
- 1a: sg_align peaked +0.345 @ w~0.33 (place_dist 0.115, best) then FADED to +0.074 @ w=0.13; place_dist crept back to 0.155.
- 1b: ended w=0 with sg_align +0.125 (below +0.2 bar), grasp COLLAPSED 1.0->0.50, place_dist back to 0.205 (transport lost), reward 0.22.
- Read: the mid-anneal recovery was REAL but TRANSIENT; it did not survive to w~=0. Learned manager is only WEAKLY load-bearing -- gains some control as the crutch weakens but cannot run the task alone; env reward did real work beyond bootstrap (grasp-maintenance + transport). 
- NUANCE: intermediate sweet spot ~w=0.3 (env+manager together best: place_dist 0.115, sg_align 0.34) but still success 0. w=0.5 -> worker freelances; w=0 -> collapse.
- COMBINED with 3a oracle 100%: worker fully capable; a GOOD manager solves; the LEARNED manager's subgoal quality is the gap. "Learned hierarchy stands alone" = rigorous PARTIAL/NEGATIVE result on single-stage PickCube.
- DECISION: proceed to TRANSFER experiment on the ORACLE source (100%-solving, embodiment-invariant); optional low-priority manager-UTD screen to push learned-manager quality.

## MILESTONE: first full solve + phase-separable signal (Exp 1/3a, ~200k) [2026-05-31]
- **3a (oracle cube-centric, w=0.5 constant) SOLVES PickCube 100%**: success 0.94@150k -> 1.00@200k/225k, reward 0.75, place_dist 0.07, sg_align 0.83. FIRST full solve of the campaign. => worker is FULLY CAPABLE; the entire gap is the LEARNED manager's subgoal QUALITY (not worker/pipeline/representation).
- **1a (learned, anneal w): PHASE-SEPARABLE signal.** sg_align faded to ~0 at w=0.5 (worker freelancing on env reward) then RECOVERED to +0.34 as w annealed 0.5->0.33 (worker re-couples to manager as the crutch is removed). place_dist kept dropping 0.13->0.115, cube@goal->0.29. Clears the +0.2 bar at w=0.33. Verdict pending w->0 (300k).
- **Learned manager is load-bearing but MEDIOCRE**: sg_align 0.34 vs oracle 0.83; place_dist 0.115 vs 0.07; success 0 vs 1.0. Becoming load-bearing != good enough.
- **place-PBRS (1b) does NOT help**: 1b transport worse than 1a (place_dist 0.18 vs 0.115).
- NEXT: (2) manager UTD bump (D2) to improve learned subgoal quality; (3) TRANSFER UNLOCKED — oracle cube-centric is a 100%-solving, embodiment-invariant, load-bearing manager => freeze it, adapt only worker on damped/weakpanda vs flat-from-scratch (interface-reuse framing).

## w=0 RESULT + Phase D reach bootstrap [2026-05-31]
Exp A/A2/B (w=0 cube-centric) ALL flat at 80k: ever_grasped=0, reward~0.06, place_dist 0.202, sg_align nan -- INCLUDING oracle B. ROOT CAUSE: at w=0 the cube-centric intrinsic is IDENTICALLY ZERO pre-grasp (hand_term=0 in cube-centric, cube_term grasp-gated off, PBRS=0 until a grasp flip) => no gradient to learn grasping, and grasping is a hard exploration event. Even the oracle can't help (its cube-subgoal is grasp-gated too). => the env-reward floor was the SOLE pre-grasp grasp-bootstrap, not just an "answer sheet"; removing it (w=0) CONFOUNDS the load-bearing test. Killed the 3 dead runs.
FIX (Phase D): added reach_pbrs_weight -- a potential-based, object-anchored pre-grasp reach term Phi=w*(1-tanh(5*||tcp-cube||)), shaping=gamma_low*Phi(s')-Phi(s), using obs tcp->obj dims (PickCube 36-38). Bootstraps grasping with NO env reward; NON-FARMABLE (hovering at cube = (gamma-1) tax, like grasp PBRS); ~0 post-grasp (gripper stays on cube); transport still ENTIRELY manager-driven => clean w=0 load-bearing test. 131 tests pass (incl. reach regression: approach +, hover tax -, off post-grasp).
RELAUNCHED (w=0, cube-centric, reach=1.0, seed1, 200k, save@50k):
  - hiro_pD_w0_reach10_cubeCentric_b00_seed1_200k        (learned mgr) -- DECISIVE
  - hiro_pD_w0_reach10_cubeCentric_pPBRS_b05_seed1_200k  (learned mgr + place-PBRS)
  - hiro_pD_w0_reach10_ORACLE_cubeCentric_seed1_200k     (oracle upper bound / worker-capability)
Milestone 1: does ever_grasped/grasp_rate lift off zero (~40-80k)? (confirms the reach bootstrap fixed the w=0 no-grasp). Milestone 2 (DECISIVE): sg_align >= +0.2 to end AND place_dist drops => manager LOAD-BEARING.

## DECISION PHASE: is the manager LOAD-BEARING? [2026-05-31]
Redirect (advisor briefing): stop chasing reward; gating metric = **sg_align** (does the worker follow the manager?). A higher reward with sg_align~0 is a NON-RESULT. 1M extension + Phase F GATED off until a load-bearing manager is confirmed on 3 seeds.
Orientation hygiene (§4): subgoal/reward/relabel metrics are POSITION-ONLY (project/intrinsic/transition/off-policy-correction/place-PBRS all use Cartesian indices; only verify_subgoal_space.py mentions quaternions, to assert indices avoid them). => no quaternion-L2 metric, no geodesic fix needed. Raw quaternions DO appear in network-input obs (obs[22:26],[32:36]); converting to 6D rep is a non-trivial obs-wrapper change, DEFERRED (orientation not the bottleneck per the 53% grasp). Exposed save_freq_steps Modal flag (screens now checkpoint every 50k for Exp C).
LAUNCHED (seed1, 200k, cube-centric, w=0, g08, no-PR, save@50k):
  - Exp A:  hiro_pA_w0_cubeCentric_b00_g08_seed1_200k       (learned manager, beta=0) -- THE decisive test
  - Exp A2: hiro_pA_w0_cubeCentric_pPBRS_b05_g08_seed1_200k (learned manager + place-PBRS b0.5, best manager-intent aid)
  - Exp B:  hiro_pB_w0_ORACLE_cubeCentric_g08_seed1_200k    (oracle = perfect fixed target, NO learned manager) -- upper-bound control
Decision rule: place_dist drops AND sg_align >= +0.2 to the END => manager load-bearing (promote to 3 seeds, THEN unlock 1M + Phase F). A fails but B works => manager-side. Both fail => worker/coupling. Exp C (frozen-manager probe) pending a high-sg_align mid-ckpt from A/A2.

## Phase 5 RESULTS — cube-centric screens (seed1, 200k, from scratch) [2026-05-31]
| metric | beta=0 | beta=0.5 (+PBRS) | old-space b00 |
|--------|--------|------------------|---------------|
| train grasp_rate | 0.00->**0.53** | 0.00->0.51 | 0.02 |
| mean_intrinsic | 0.027->0.091 | 0.027->0.092 | ->0 (hover) |
| place_dist | 0.20->**0.115** | 0.20->0.124 | 0.23 flat |
| cube_at_goal | 0.06->0.20 | 0.06->0.18 | 0.03 |
| eval/reward | ->**0.467** | ->0.439 | 0.17 |
| success | occ. 1/16 (late) | 1 blip | 0 |
| sg_align | +0.30 mid -> ~0 late | +0.50 mid -> -0.35 late | noisy |

VERDICT: **HOVER-HACK DIAGNOSIS CONFIRMED + FIXED.** Cube-centric raised grasp_rate 2%->53% (~25x) and mean_intrinsic rose WITH grasp (the predicted signature: no hand-term => intrinsic only via grasp+cube-move). Best transport of the campaign (place_dist 0.20->0.115, reward 0.47) -- the thesis object-centric representation doing real work. NOT yet a full solve (success occ. 1/16; cube stalls ~11cm out, not held static; sg_align faded late so worker+env-reward carries much of the transport). KEY: STILL IMPROVING at 200k (reward & place_dist both still moving), NOT plateaued => under-trained, not stuck. Next: extend cube-centric (beta=0, the best) to ~1M (Phase 3 length test) to see if it crosses the 0.6 ridge / lifts success.

## Phase 5 BUILT (cube-centric grasp-gated subgoal) [2026-05-31]
Code: CUBE_CENTRIC_SUBGOAL_SPACES (PickCube indices [29,30,31], object_dims=(0,1,2) => hand_positions empty => hand_term==0 in phased_pbrs); get_subgoal_space(env,mode); trainer/modal subgoal_mode flag; make_oracle(cube_only=) for an oracle sanity check. Worker earns intrinsic ONLY via grasp PBRS + grasp-gated cube term; pre-grasp guidance from hybrid env reward. Tested: 128 pass incl. hover-hack regression (not-grasped gripper motion -> reward 0 in cube-centric vs farmable in old space) + end-to-end 3D-subgoal agent (te_high=-8.69). READY to launch as a from-scratch screen (subgoal_mode=cube_centric, w=0.5, g08; optionally + place_pbrs_beta).
