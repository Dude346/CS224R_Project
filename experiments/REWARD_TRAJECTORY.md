# HIRO worker-reward trajectory (PickCube-v1)

This document records how the **low-level (worker) reward** for our HIRO agent
evolved across the project, what each version was trying to fix, the exact math
we used, and — just as important — where each version was still wrong. It covers
only our lineage (`main` → `hiro-reward-modified` → `latent-object-hiro-fixes`).

The headline: **none of these were RL-algorithm failures. They were reward
mis-specification.** SAC optimized faithfully every time; it just optimized the
wrong thing, because dense shaping rewards leak. Each fix revealed the next leak.

---

## 0. What we are shaping *toward*: the environment reward

The worker reward only matters relative to the task's true success condition, so
everything below is best read against PickCube's own dense reward
(`mani_skill/.../pick_cube.py`):

```
reaching = 1 − tanh(5 · ‖tcp − obj‖)
reward   = reaching
reward  += is_grasped                                   # +1 when grasped
place    = 1 − tanh(5 · ‖obj − goal‖)
reward  += place · is_grasped                            # transport, gated by grasp
static   = 1 − tanh(5 · ‖arm_qvel‖)                      # arm joints only (qvel[:-2])
reward  += static · is_obj_placed                        # settle, gated by "placed"
reward[success] = 5                                      # success overrides
normalized_reward = reward / 5
```

Success (`pick_cube.py::evaluate`) requires **both**:
- `is_obj_placed`: `‖obj − goal‖ ≤ goal_thresh` (default **0.025 m**), and
- `is_robot_static`: arm `‖qvel‖ < 0.2`.

Two facts that drive the entire story:
1. **The goal is in the air.** Its z is randomized in `[obj_z, obj_z + 0.3]`, so
   "place" almost always means **lift**.
2. **Max normalized reward is 1.0** (raw 5). A policy that reaches + grasps but
   never transports tops out around `(1.0 + 1.0)/5 = 0.40`.

### The diagnostic that cracked the hardest bug
Our broken agent **plateaued at eval reward ≈ 0.37 (raw 1.85), success = 0.**
Decomposing against the formula above:

```
1.85 ≈ reaching(1.0) + is_grasped(0.85) + place(≈0) + static(0)
```

i.e. **it grasps and holds the cube but never lifts it.** Reading the *plateau
value* as reward components — not staring at the loss curves — is what localized
the bug to the reward, not the algorithm.

---

## 1. Core concepts (used at every step)

**HIRO subgoal transition (telescoping target).** The manager emits a subgoal
`g` at state `s₀`; the absolute target is `T = proj(s₀) + g`. As the worker
moves, the *remaining* displacement is tracked by
```
subgoal_transition(s, g, s′) = proj(s) + g − proj(s′)          # = g′, the residual to T
```
`proj(s) = s[indices]` selects the subgoal dims (e.g. `tcp→obj`, `obj→goal`).
The worker has "reached" the subgoal when this residual is 0.

**Shaping reward.** A dense per-step signal that points toward success, because
the true success reward is far too sparse to learn from directly.

**Farming.** If any per-step term can be collected **without** advancing the
task, the optimizer will collect it indefinitely instead of doing the task. This
is the villain in R0, R1, and R3.

**Potential-based reward shaping (PBRS), Ng, Harada & Russell (1999).** Pick a
potential `Φ(s)` and pay only its discounted change:
```
F(s, s′) = γ · Φ(s′) − Φ(s)
```
**Theorem:** adding `F` to any reward does **not** change the optimal policy. The
consequence we care about: a telescoping potential **cannot be farmed** — any
trajectory that returns to a state nets `≈0` from `F`, so there is no exploitable
cycle. Standing still pays `(γ−1)·Φ`; with `γ = 1` it pays exactly 0.

These concepts recur verbatim below; the whole arc is essentially us re-deriving
"make every shaping term a potential" the hard way.

---

## 2. The trajectory

Notation: `hand` = hand/reach dims of the residual (`tcp→obj`), `obj` = object
dims (`obj→goal`), gated by grasp. `igs = is_grasped(s)`, `igs′ = is_grasped(s′)`.
All of these are the **worker** reward only; the env reward is never modified.

---

### R0 — Plain HIRO distance
`SubgoalSpace.intrinsic_reward`

**What we were doing.** Textbook HIRO: pay the worker the negative distance to
the manager's target, over the full subgoal `[hand_xyz, cube_xyz]`.
```
r = −‖ proj(s) + g − proj(s′) ‖₂ = −‖residual‖₂
```

**The problem.** Pre-grasp, **the worker cannot move the cube**, so the cube part
of `‖residual‖` is an irreducible penalty. Because it is a *single Euclidean
norm* over `[hand, cube]`, the unreachable cube error is not separable from the
hand error — it distorts the gradient on the dimension the worker *can* control
and makes the reward scale uninformative. (StackCube is worse: the second cube
never moves → a permanent negative floor. This is "the pathology that blocked the
early runs.")

**What it intended vs. what it missed.** Intended: a generic, task-agnostic HIRO
signal. Missed: that a subgoal dimension which is *physically uncontrollable in
the current phase* poisons the reward rather than simply being ignored.

**Lesson → next step.** Don't include dims the worker can't currently affect.
Gate the reward by phase (grasped vs. not).

---

### R1 — Phased reward + discrete grasp bonus
`SubgoalSpace.phased_intrinsic_reward`

**What we were doing.** Split by phase. Before grasp: only the hand term. After
grasp: add the cube term. Plus a one-time bonus the step the grasp is acquired
(`just_grasped` = grasp flips F→T):
```
r = −‖residual_hand‖ + igs · (−‖residual_obj‖) + just_grasped · grasp_bonus
```

**The problem.** `just_grasped · grasp_bonus` is a **discrete event reward**. The
worker learned the cycle **grasp → release → grasp → release …** to re-collect
the bonus. Textbook farming.

**What it intended vs. what it missed.** Intended: remove the unreachable-dim
penalty (succeeds) and nudge the worker to grasp. Missed: that "+reward for
doing X" with no cost to *undoing* X is always farmable by doing/undoing X.

**Lesson → next step.** Make the grasp incentive **un-farmable** — a potential,
not an event bonus.

---

### R2 — Phased + PBRS grasp potential (true grasp bit)
`SubgoalSpace.phased_pbrs_intrinsic_reward`

**What we were doing.** Replace the discrete bonus with a grasp **potential**
`Φ_grasp = α · is_grasped`, paid as PBRS. Also switched from a hand-rolled
positional grasp heuristic to the **environment's true grasp bit** `obs[18]`
(commit *"Use true PickCube grasp bit"*), so the signal can't be spoofed by
hovering at the right geometry.
```
r = −‖residual_hand‖ + igs · (−‖residual_obj‖) + ( γ·α·igs′ − α·igs )
```
The cube term is gated by `igs` (grasp at the state the action was taken in), not
`igs′`, so the grasp-acquisition step isn't charged for "far cube," and a drop
pays the residual-at-drop.

**Why the PBRS term works.** A drop→regrasp cycle nets `α(γ−1) < 0` — farming is
mathematically dead. (`γ` here must equal the worker discount `gamma_low` for
exact policy-invariance.)

**The problem / what it missed.** The hand and cube terms are still **raw
negative distances** (`r ≤ 0` always). Pure-penalty shaping tells the worker
"everything hurts, hurt least," which biases it toward **passivity** — moving
risks a larger penalty, so doing little looks safe. We got reliable
reach+grasp, but it stalled on the effortful positive behavior (lifting the cube
up to the aerial goal). Also, the `γ<1` PBRS form leaves a small per-step
**"hold-tax"** `α(γ−1)` for merely *holding* the grasp — negligible at small `α`,
but the same mechanism becomes a real bug at R3/R4 scale.

**Lesson → next step.** PBRS is the right tool; penalties under-motivate the
positive behaviors. Switch distance penalties to **progress** (telescoping
gains), and add a signal that actively pulls the worker toward the cube/goal.

---

### R3 — Progress shaping + dense reach bonus  ⟵ the headline failure
`SubgoalSpace.phased_progress_pbrs_intrinsic_reward` (latent-era; since removed)

**What we were doing.** Two changes — one genuinely good, one fatal.

- **Good:** hand/cube penalties → **progress** terms (pay the *reduction* in
  residual this step; telescoping, can be positive):
  ```
  hand_progress = ‖g_hand‖ − ‖g′_hand‖
  obj_progress  = ‖g_obj‖  − ‖g′_obj‖
  ```
- **Fatal:** to bootstrap reaching while the manager was still random, add a
  **dense, always-on reach reward** plus anti-hover guard penalties:
  ```
  dense_reach   = reach_coef · (1 − tanh(reach_temp · ‖tcp→obj(s′)‖))   # reach_coef=1, temp=10
  hovering      = (‖tcp→obj(s′)‖ < 0.04) AND not grasped(s′)
  stall         = hovering · 0.05
  open_penalty  = hovering · 0.08 · clamp(gripper_aperture/0.04, 0, 1)
  pbrs          = γ·α·igs′ − α·igs                                       # γ=0.8, α=0.1

  r = hand_progress + dense_reach + igs · obj_progress + pbrs − stall − open_penalty
  ```

**The problem (precisely).** `dense_reach` is **ungated by grasp and is not a
potential**: it pays `≈ +1` every step the gripper is on the cube — grasped or
not, lifting or not. Consequences:
- **Grasp-and-hold local optimum.** Grab the cube, keep the gripper on it,
  collect `≈ +1/step` forever. No lifting required.
- **Signal swamping.** The lift signal is `obj_progress ≈ 0.01–0.02/step` (a
  telescoping term). Beside a `≈+1/step` constant, the *advantage* of lifting is
  ~1% of the per-step reward — **there is effectively no gradient to lift.**
- The `stall`/`open_penalty` guards only fire **pre-grasp and near**, so once the
  cube is grasped they contribute nothing; they were band-aids over a hole they
  didn't cover.

**Evidence.** Eval reward flatlined at **0.37 normalized**, success **0** — which
decodes (see §0) to **reaching ≈ 1, grasp ≈ 0.85, place ≈ 0**: grasps and holds,
never transports. The measured plateau matched the predicted exploit exactly.

**What it intended vs. what it missed.** Intended: progress shaping (good) plus a
dense kickstart for reaching. Missed: that a single **ungated, non-potential**
per-step term doesn't just add bias — it **flattens the reward landscape around
the exploit**, hiding the real task from the optimizer. We had spent R1–R2 making
the reward farm-proof and then re-introduced a farmable term right next to it.

**Lesson → next step (the project's key insight).** *Every* shaping term must be a
telescoping potential. No exceptions, no "just one dense bootstrap term."

---

### R4 — Telescoping task potential (the fix)
`SubgoalSpace.task_potential_intrinsic_reward`

**What we were doing.** Delete `dense_reach` and the guard hacks. Make **every**
term a telescoping potential `Φ(s′) − Φ(s)`, where `Φ` **mirrors the env's own
success recipe** (reach + grasp + place). Use **`γ = 1`** for the telescope (a
pure difference) to remove the hold-tax entirely.

```
Φ(x) = reach_coef · (1 − tanh(k · ‖tcp→obj(x)‖))                 # reach_coef = 1
     + grasp_coef · is_grasped(x)                                # grasp_coef = 1
     + place_coef · is_grasped(x) · (1 − tanh(k · ‖obj→goal(x)‖))# place_coef = 1,  k = 5

r = [ Φ(s′) − Φ(s) ]                          # task shaping, γ = 1
  + ( ‖g_hand‖ − ‖g′_hand‖ )                  # follow the manager's hand subgoal
  + igs · ( ‖g_obj‖ − ‖g′_obj‖ )              # follow the manager's place subgoal
```

**Why this is correct.**
- **Farm-proof by construction.** Every term telescopes → any loop nets 0
  (Ng 1999). There is no exploitable cycle.
- **No hold-tax.** With `γ=1`, holding still pays exactly `Φ(s)−Φ(s)=0`, so the
  success state is never punished (this was the latent danger from R2).
- **Correct incentives.** Holding the cube → `Φ` unchanged → **0**. Lifting the
  cube toward the goal → `place` term rises → **positive**. Dropping → `Φ` falls
  → **negative**. Therefore **lifting strictly beats holding** — the exact
  property R3 violated. We locked this in with a unit test
  (`test_lifting_beats_holding_post_grasp`): smoke run gave lift `+0.24` vs hold
  `0.0` through the full latent decode path.

**What it intended vs. what it missed (the real tradeoff).** To *guarantee*
solvability, `Φ` is built from the **true** `tcp→obj` and `obj→goal` distances
(task-aligned), **not** the manager's subgoal residual. So the dominant
task-shaping signal **partly bypasses the manager** during the lift phase — the
worker becomes a bit more of a flat shaped agent and a bit less of a pure HIRO
subordinate. We deliberately accepted *"HIRO purity ↓, reliably-solves-task ↑,"*
and kept the manager-conditioned progress terms so the hierarchy still steers
reaching/placement. What R4 alone still **missed**: it rewards *lifting to the
goal* but not *coming to rest there* — and PickCube success also requires
`is_robot_static`. So R4 can hover the cube at the goal without ever satisfying
the static condition. That gap is what R5 closes.

---

### R5 — Adding "settle", transport weighting, and HER
`SubgoalSpace.task_potential_intrinsic_reward` (current) + worker HER

**What we were doing.** Extend `Φ` with the env's **last** success component (a
static/settle bonus), up-weight transport, and densify learning with
hindsight relabeling.

```
Φ(x) += settle_coef · is_placed(x) · (1 − tanh(k_v · ‖arm_qvel(x)‖))
        where  is_placed(x) = [ ‖obj→goal(x)‖ ≤ goal_thresh ]      # goal_thresh = 0.025
               k_v = 5,  settle_coef = 1,  arm_qvel excludes the 2 gripper joints

r = [ Φ(s′) − Φ(s) ]
  + ( ‖g_hand‖ − ‖g′_hand‖ )
  + igs · object_progress_coef · ( ‖g_obj‖ − ‖g′_obj‖ )           # object_progress_coef = 2
```

**Design points / what each piece solves.**
- **Settle term** mirrors `static_reward · is_obj_placed` in the env. It is gated
  on the **discrete** `is_placed` (cube actually inside the success tolerance),
  not on grasp or the smooth place score, so it only switches on where success is
  achievable and then rewards the arm coming to rest — closing R4's gap.
- **Arm-only velocity** (`qvel[:-2]`, mirroring the env): the gripper fingers
  keep jittering while gripping the cube, so including them would make
  "be still" unreachable and the static signal could never reach 0.
- **`object_progress_coef = 2`** up-weights following the manager's transport
  subgoal relative to reach, since lifting is the bottleneck behavior.
- **HER in the worker** (`low_her_ratio = 0.8`): the telescoping potentials are
  farm-proof but give weak *early* signal; hindsight-relabeling worker
  transitions with achieved subgoals densifies the learning signal honestly.

**Tradeoff / what to watch.** More terms = more knobs = more ways to mis-balance.
The discipline that keeps it safe: **every term is a potential that mirrors a
real component of the env's success condition** (reach, grasp, place, static).
We add nothing that can be collected without moving toward success. Open risk:
the discrete `is_placed` gate makes `Φ` jump at the tolerance boundary, so the
settle bonus appears/disappears sharply near the goal — worth monitoring for
chattering.

---

## 3. Recurring tradeoffs (the through-line)

| Tension | One end | Other end | Where it bit us |
|---|---|---|---|
| **Reachability vs. completeness** | full goal in reward (R0) | phase-gated dims (R1+) | R0's uncontrollable-cube penalty |
| **Bootstrapping vs. farmability** | dense always-on reward (R3) | telescoping potentials (R4) | R3 grasp-and-hold; needed HER (R5) to densify *without* a farmable term |
| **Penalty vs. progress vs. potential** | distance penalty (R0–R2) | telescoping potential (R4–R5) | R2 passive worker; progress-but-mixed (R3) still farmable |
| **HIRO purity vs. reliability** | manager-conditioned reward | task-aligned reward | R4 task term partly bypasses the manager |
| **Smooth vs. discrete gates** | smooth `1−tanh` | discrete `is_placed`/`is_grasped` | R5 settle bonus can chatter at the tolerance edge |

---

## 4. Meta-lessons for the writeup

1. **The bottleneck was reward specification, not optimization.** SAC was fine at
   every step. We had ~5 reward rewrites because dense shaping rewards leak: the
   agent exploits any per-step term that doesn't require task progress.
2. **The fixes were monotonic in one principle.** Every failure was a violation of
   potential-based shaping (a discrete bonus R1, a `γ<1` hold-tax R2, an ungated
   dense term R3). The final rule — *every shaping term is a telescoping potential
   mirroring a true success component* — is one sentence, but it took five
   concrete failures to earn it.
3. **Diagnose by decoding the plateau, not by reading losses.** The 0.37 plateau
   → "reach+grasp, no place" decomposition is what localized the R3 bug.
4. **Hierarchy adds its own axis.** Whether `Φ` is task-aligned or
   manager-conditioned trades HIRO faithfulness against reliability; we blended
   both and were explicit about which dominates.

---

## 5. Status & numbers to fill in

| Version | Code symbol | Measured result |
|---|---|---|
| R0 | `intrinsic_reward` | qualitative: uncontrollable-dim penalty (not run to convergence) |
| R1 | `phased_intrinsic_reward` | qualitative: grasp-bonus farming |
| R2 | `phased_pbrs_intrinsic_reward` | reach+grasp learned; lift stalls *(fill in success/return)* |
| R3 | `phased_progress_pbrs_intrinsic_reward` | **eval reward ≈ 0.37 (raw 1.85), success = 0** — grasp-and-hold |
| R4 | `task_potential_intrinsic_reward` (γ=1, reach+grasp+place) | unit-verified (lift +0.24 > hold 0.0); *full-run numbers TBD* |
| R5 | `task_potential_intrinsic_reward` + settle + HER | *in progress* |

**TODO for the paper:** drop real W&B `success_once` / `return` numbers for R2,
R4, R5 from `eval_curves.csv` on the `cs224r-project-results` Modal volume. Only
R3's 0.37 plateau is a hard measured number so far.
