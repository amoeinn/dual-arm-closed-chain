# dual-arm-closed-chain

Two arms on a free-flying servicer berthing a station module, where closing the
second gripper turns the system from a tree into a closed kinematic loop and
every object built for the one-armed case stops describing it. In-space
assembly is the driving case; the repository is named for the claim because
that is what makes the problem different.

![Two Panda arms on a free-flying bus, holding a 0.40 m module between them at the grasp pose](docs/servicer.png)

The bus carries mass and inertia and no geometry, because it is a radius of
gyration rather than a shape. The translucent box above is a proxy drawn for
this picture only and takes no part in any measurement.

**Every figure here is drawn from a quantity the test suite asserts**, so a
figure cannot silently disagree with the text around it. Where a figure and a
test disagree, the test is right and the figure is stale. One of the three
figures below found a defect in a result this project had already published,
which is recorded under Bugs rather than quietly fixed.

## What is here

**The claim.** A servicer with one arm is a tree, and stays a tree. Two arms
berthing a module is a tree until the second gripper closes and a closed
kinematic loop afterwards. Every object Project C built assumes a tree, and the
task this project exists to perform is defined by crossing out of that
assumption, at the instant the task succeeds.

The assumption is not implicit. The mass matrix is built as a sum over
per-link centre of mass Jacobians whose columns are set by ancestry, and
ancestry is a tree concept. Close a loop and a link's motion is no longer
determined by its ancestors alone, so `H_b`, `H_bm`, the generalized Jacobian
`J_g` and the momentum-conservation argument underneath all of them are
computed on a model that no longer describes the system.

**A generated two-arm model.** Two Panda chains, name-prefixed per arm, on a
free-floating bus. Two Pandas rather than a bespoke servicer arm is deliberate:
it reuses Franka's published identified inertial parameters, which Project C
verified against PyBullet's free-base mass matrix to 5.7e-09 and against an
independent C++ implementation to 3.553e-15. A bespoke arm would mean new
inertial parameters with no independent check.

Nothing may be measured before the model is asserted, so seven properties are
checked on every generation: that it parses as a URDF, that both arms are
reachable from the bus under names shared with neither, that total declared
mass is the bus plus one arm per arm, that every inertia tensor survives
composition with its off-diagonal terms intact, that every joint keeps its own
limits, that **a consumer builds the inertia the file declares**, and that the
grasp pose sits inside every joint limit. The sixth exists because PyBullet
ignores declared inertia by default and built the bus with 0.00013 kg m^2
against the 50.0 declared, wrong by a factor of 375,000, silently, past five
checks that only read the file.

**Grasp geometry, and which of it is derived.** The gripper opens 0.0800 m and
the module is 0.4000 m across, so the grasp is on fixtures at the module's
faces rather than on the module. That is load bearing twice: hand separation at
grasp is `0.4000 + 2 x 0.1142 = 0.6285 m`, which sets the mount spacing at
+/- 0.8000 m, and the loop closes through the fixtures, which sets the
constraint equations. The grasp height of 0.55 m and the fore-aft envelope are
**stated choices with no berthing port specification behind them**, and every
margin quoted here is conditional on them.

**Vendored dynamics, generalised to N chains.** `kinematics.py` and
`dynamics.py` are copied from Project C at commit `343e2152`, with a provenance
record naming every intentional deviation. `kinematics.py` needed no change at
all. `dynamics.py` needed less than expected: the per-link ancestor filtering
was already there, so the mass matrix and coupling inertia generalise nearly
for free, and the one real blocker was a method that walked a single chain.
The generalisation is held to Project C's committed numbers on Project C's own
single-arm model, and the assertion that carries it is the closed-loop base
rotation of 18.98 degrees, one scalar exercising the whole stack.

**A four-layer reference, built before the formulation it checks.** Momentum
conservation, the open-chain limit, a second solver, and constraint forces
doing no work. The second solver was meant to be PyBullet's and cannot be: it
does not hold this closure. What replaces it is the constrained solve written
twice, Lagrange multipliers against analytic projection, agreeing to 2.386e-14
relative.

## Results

### What the second arm buys

`H_bm` maps joint rates to base momentum. Its null space is the set of joint
motions that disturb the base not at all, and with one 7-DOF arm against a
6-DOF base that null space is one dimensional: a reactionless direction exists,
but nothing is left over to also satisfy a task. With two arms it is eight
dimensional.

Eight is not the number that matters. What matters is that **either hand has
all six task directions available while the base stays exactly still**, in 200
of 200 sampled configurations, against one of six for a single arm. That is the
difference between a reactionless direction existing and reactionless motion
being a constraint a berthing trajectory can be planned under.

`H_bm` was rank 6 in 300 of 300 samples for both models. Rank loss was hunted
rather than assumed away, by descent on the smallest singular value from many
starts inside the joint limits, because a uniform sample over a 14-dimensional
box would never land on a measure-zero set and reporting that it did not would
say nothing. Forty restarts of four hundred steps found none. The search drove
one arm from a sampled median of 2.15e-02 down to 5.26e-06, a factor of 4000,
and two arms from 2.11e-01 only to 3.35e-03, a factor of 63, so the two-arm
coupling is markedly harder to push toward singular. **Not reachable by this
search** is the claim, not that no such configuration exists.

A limitation with the eight: the image of the null space in the twelve
dimensional two-hand task space is eight dimensional, so both hands cannot be
commanded independently and reactionlessly at once. Six for one hand, or eight
of the twelve shared.

### What the grasp transition costs

Two questions, and only the second bears on a controller. Whether the plant
changes is arithmetic and always yes. Whether anything notices is the one worth
measuring: take a joint motion that is exactly reactionless on the pre-grasp
plant, verified to 2.3e-18, apply it to the post-grasp plant, and read the base
twist it now produces against what an arbitrary joint motion produces there.

![Base disturbance from a pre-grasp reactionless command, applied after the grasp, against module-to-servicer mass ratio](docs/grasp_transition.png)

Median over twenty draws from the eight-dimensional null space: 0.050 at a
mass ratio of 0.002, 0.236 at 0.01, 0.799 at 0.04, 1.180 at 0.08, 1.253 at
0.17, **1.276 at 0.35**, then 1.154 at 0.7, 1.044 at 1.5, and about 1.03 for
two further decades.

**The curve peaks and settles rather than saturating, and the worst case is a
moderate module rather than a heavy one.** The peak sits at a ratio of 0.35,
about 82 kg, where a pre-grasp reactionless command produces 28 per cent more
base disturbance than an arbitrary joint motion would. A module twenty times
the servicer's mass is *less* damaging in relative terms.

The reading is that a heavy enough module dominates the plant for every command
alike, so a pre-grasp command stops being specifically mis-aimed and becomes
merely uninformed, which an arbitrary command already was. At moderate ratios
it is actively pointed the wrong way: it was optimised to cancel a base
reaction the module has since changed, and cancelling the wrong thing is worse
than not trying.

Below a ratio of about 0.01, roughly two kilograms, the transition is a
perturbation. The 40 kg module this project carries sits at 0.17, close to the
peak: by `H_b` it is the 20 per cent perturbation it was chosen to be, and by
the measure a controller feels it dominates. The three measures disagree about
the same module.

### What the tree formulation returns when the loop closes

Registered as a prediction before any of this was built: that Project C's
`generalized_jacobian`, run on a closed loop, returns a plausible number rather
than raising. **It holds**, and it is the first silent failure in this
portfolio predicted in advance rather than found afterwards.

It does not raise. It returns a 6 by 14 tensor, every entry finite, rank 6,
with a norm of 2.8300 against the fixed-base `J_m`'s 2.8329. Right shape, full
rank, right order of magnitude: it survives every check a caller would make.

That it does not raise is a statement about error handling. What it is wrong by
is the useful claim. Measured against the composite plant over 24 admissible
unit joint rates with the 40 kg module, the **relative error is a median 22.3
per cent**, range 6.5 to 119.0, and the **direction is out by a median 11.4
degrees**, worst 64.85.

Twenty two per cent is the dangerous size: too large to be right, too small to
look wrong, and it reads as modelling error or integration slop rather than as
a bug. The direction error rules out the comfortable reading, since a pure
scale factor could be calibrated out of a controller and an 11 degree median
heading error cannot.

The whole discrepancy is in the base twist. `J_m` and `J_b` are geometry and
the closure does not touch them; what the tree formulation gets wrong is
`H_b^-1 H_bm`.

### Why the model stops being valid at the moment the task succeeds

The free-floating formulation assumes no external wrench. Contact injects one,
so the model is valid up to the instant the task succeeds and not through it.
The two regimes are measured separately and the boundary stated, rather than
integrating a trajectory across a discontinuity.

![Servicer displacement accumulating over the carry, on the plant after the grasp and on the plant before it](docs/mate_drift.png)

**Before contact.** Carrying the 40 kg module 0.40 m forward displaces the
servicer **50 mm**, against **21 mm** predicted on the pre-grasp plant: a
factor of **2.4**. The per-step error of 22.3 per cent compounds rather than
cancelling, because it is a systematic bias and not noise.

**These figures are a correction, and the earlier ones are withdrawn.** This
result was first published as 105.84 mm against 25.36 mm over a 0.50 m carry,
a factor of 4.2. Both numbers were contaminated. The original measurement was
reading a **solver posture flip rather than servicer drift**: each waypoint was
solved independently with the palm roll swept afresh, the winning roll changed
between neighbours, and the joint vector jumped 4.876 rad in a single step
against 0.11 either side, which integrated into about 56 mm of base motion
belonging to the inverse kinematics. Solving it continuously also showed that
**the 0.50 m carry was never continuously achievable**: across all twelve rolls
the longest reachable without a posture change is 0.40 m, and 0.50 m fails by
34.15 mm. The original run described a trajectory that does not exist.

**The boundary.** Before contact the base twist is set by momentum
conservation. After contact the module is fixed to the station, so it is set by
kinematics instead: the hands must stay on a body that no longer moves and the
arms push off it. These are different laws, not one law with a correction term,
and no interpolation between them is meaningful.

**After contact.** Base twist per unit joint rate over 24 random directions is
a median 0.0552 free floating against 0.5178 berthed, a ratio of **9.44x**
ranging 5.38 to 21.72. A controller carrying the free-floating model through
contact would be wrong about the base response by an order of magnitude, and
wrong in a way no single gain retunes, since the ratio varies fourfold across
directions.

## Bugs found and fixed

Sixteen, and what they have in common is worth naming before the list. Not one
was a crash, and not one was found by reading code. Every one produced a
plausible number that survived the checks a caller would make, and every one
was found by two things that should agree failing to, or by a control that
should have been trivial coming out wrong.

They group into four families, and the grouping is more useful than the count.

### The same frame distinction, five times

PyBullet reports a centre of mass where a link frame is meant, and the two are
31.62 mm apart on `panda_hand`.

`getLinkState` index 0 is the link's centre of mass and index 4 is the link
frame. Comparing an inverse kinematics result against index 0 returned exactly
31.62 mm for every target, which read as a solver that never converged. It had
converged every time.

`createConstraint` takes `parentFramePosition` in the link's centre of mass
frame. Passing `[0, 0, 0]` while computing the child offset from the link frame
built a 31.62 mm violation into the closure: **31.6 kN of constraint force on a
system at rest**, and a scene that left in three metres.

The guard watching that closure measured a parallel quantity, hand-to-centre
distance against a nominal, and **read 0.00001 mm while the constraint it was
watching was 31.62 mm out**. A guard that measures something else reports that
everything is fine.

And the momentum sum twice: once taking the link frame Jacobian where the
centre of mass one was needed, and once more when the correction was applied to
the wrong argument.

### Two false accusations, which cost more than the real ones

Twice this project concluded that `calculateJacobian` was unusable and should be
replaced. It is correct. Given a `localPosition` of `[0, 0, 0]` it returns the
Jacobian of the link **frame** origin, and against a central difference of that
same point it agrees to 2.491e-05, which is the finite-difference floor in
Bullet's single precision. It reads 1.239e-02 against the centre of mass
because that is a different point. Both accusations were that comparison.

These cost more than the genuine bugs. A real bug gets fixed; a wrongly blamed
dependency gets replaced by something home made, which is what happened here
before the comparison was checked. The rule the project now carries: before
concluding a library is wrong, establish which point, frame and convention its
answer is expressed in.

### Guards that read as protection and were not

The assertion that two chains agree about a shared joint's world frame is
**unreachable** on the two-arm servicer, because the arms share no joints. It
had never fired because it could not, which is not the same as passing. It is
now tested on a topology where it can fire, wrist and elbow on one arm.

`_assert_base_convention` was **checking the wrong arm**. Generalising the
end-effector API added an `arm` parameter to the caller and not to the guard,
so asking for arm 1 validated arm 0 and then used arm 1. The verdict happened
to be the same, so nothing was wrong, but the guard was no longer checking the
object it guards.

And the reach margin table, five poses at a 0.00 mm residual, was meaningless:
`calculateInverseKinematics` ignores joint limits unless `lowerLimits`,
`upperLimits`, `jointRanges` and `restPoses` are **all** passed, and four of
fourteen joints sat outside their range, two by more than 1.5 rad. Every margin
described free space rather than a posture the arm could hold.

### Six that stand alone

**PyBullet ignores declared inertia** and derives its own from collision
geometry. The bus has none, so it was built with 0.00013 kg m^2 against the
50.0 declared, wrong by a factor of 375,000, silently, past five model checks
that only read the file.

**PyBullet's constraint solver cannot hold this closure.** A system at rest,
zero torque, zero velocity, zero closure error, gains 15 kJ and travels 1.65 m.
More solver iterations and shorter timesteps make it worse, which is not a
convergence signature. That is why the second solver was dropped and the
constrained solve written twice instead. The control that established it, a
system at rest must stay at rest, should have been the first thing run and was
not.

**Finite differences at 1e-6 are mostly noise** when the library computes in
single precision. Base columns that must be exactly the identity came back as
1.013. Central differences at 1e-3 give 1.000002, and the error grows as the
step shrinks, which is the signature of cancellation rather than truncation.

**Velocity control over-constrains a closed loop.** Commanding fourteen arm
rates independently describes a motion a closed loop cannot perform, and the
solver resolves the contradiction with impulses: 16 kJ of kinetic energy at the
first step against 0.07 kJ for the same command with the loop open. Nothing was
wrong with the solver. The command was inconsistent, which is this project's
own claim arriving early and in the test harness.

**Position-only inverse kinematics put both forearms inside the payload.**
Constraining hand position and leaving orientation free let the solver pick a
wrist that folded link 5 and link 6 **74 mm into the module the hands are
constrained to**. The pose reached its targets exactly and was physically
impossible. Fixing the palm orientation resolved it, and only the hand and
finger links are now filtered from collision so that an unintended contact
anywhere else stays visible.

**A figure caught what a total did not.** The mate result was published as
105.84 mm of servicer displacement. That number looks unremarkable. Plotted, the
curve had a near-vertical step in it at one waypoint, rising 57 mm to 105 mm,
and the cause was the inverse kinematics changing posture branch: the joint
vector jumped 4.876 rad in a single step against 0.11 either side. About 56 mm
of the published figure belonged to the solver. The corrected value is 50 mm
over a carry that is actually achievable.

That is the case for plotting a result rather than tabulating it, and it is
worth stating plainly: **a total hides its own shape**. Nothing about 105.84 mm
invites suspicion. A curve with a cliff in it invites nothing else.

## Tests

71 tests in about 44 seconds. Every one is named for the invariant it protects
rather than the function it calls, and **every one is paired with the specific
break it exists to catch**, because a check that has never been shown to fail
has not been shown to work.

That discipline earned its cost twice in this project, and in both cases the
mutation found the test rather than the code. One guard turned out to be
unreachable on the model it shipped with. One mutation that was expected to
surface as a disagreement turned out to be a change of gauge: flipping the sign
of `J^T` in the saddle point matrix only flips the multipliers, so the
accelerations came back 1.9e-13 apart, and the test written to catch it could
never have fired.

The suite also carries what the pair of derivations **cannot** check, asserted
rather than described: both take the same mass matrix and constraint Jacobian,
so corrupting either leaves them agreeing to 1e-14 and both wrong.

Four things are deliberately left as scripts rather than promoted. The
rank-loss search, at 295 seconds, whose result is a negative from a search and
not an invariant; the thirteen-point mass ratio sweep, whose two ends are
asserted and whose middle is a curve; the 200-step mate, which the tests do in
20 at the same conclusion; and the mount spacing sweep, whose output is already
pinned by a model property.

## Limitations

**No contact dynamics anywhere.** The two regimes either side of contact are
measured and the boundary is characterised, but nothing integrates through it.
What contact does to the model is stated exactly; what it does during is not
modelled at all.

**No controller.** Every result here says what a plant does, not whether
something could fly it. That the pre-grasp model is worthless after the grasp
is measured; that a berthing would fail is not claimed.

**The grasp height and the fore-aft envelope are stated choices.** There is no
berthing port specification behind `z = 0.55 m` or the envelope, so every reach
margin is conditional on them. Given a port they become derived and the spacing
is re-run rather than edited.

**The grasp is on fixtures, by necessity.** The gripper opens 0.0800 m against
a 0.4000 m module. Where those fixtures sit sets the loop geometry, so a
different grasp scheme changes the constraint equations and not just the
numbers.

**The fore-aft envelope is asymmetric**, and the carry is bounded by posture
rather than by reach. Forward reaches +0.60 m, aft does not reach -0.60 m, and
a continuous carry without a posture change runs to 0.40 m, not further.

**Rank loss was not found, which is not the same as absent.** Forty restarts of
four hundred descent steps found no singular `H_bm` inside the joint limits. A
harder search might.

**Single precision is the floor on the constraint Jacobian.** It is
differenced from positions because Bullet computes in floats, so layer A's
closed-loop residual sits at 1e-06 rather than at machine precision, and that
floor is measured rather than assumed.

## Requirements

Python 3.12, and the pins in `requirements.txt`: torch 2.14.0+cpu, pybullet
3.2.7, numpy 2.5.2, matplotlib 3.11.2. `pytest 9.1.1` is in
`requirements-dev.txt`. Everything runs on CPU.

**MoveIt is required, and not for a ROS node.** See Running it below before
starting.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

If ROS 2 is sourced in your shell, unset `PYTHONPATH` before running anything.
pytest autoloads plugins from every `site-packages` on the path, finds a
ROS-installed one, and fails with `ModuleNotFoundError: No module named 'yaml'`,
which names `yaml` and says nothing about ROS.

```bash
PYTHONPATH= .venv/bin/python -m pytest tests/
```

## Running it

**Read this before the first command.** The dual-arm model is generated, not
committed, and the chain behind it reaches outside this repository:

```
MoveIt's moveit_resources_panda_description  (panda.urdf.xacro)
  -> Project C expands it to panda_identified.urdf
    -> this project composes two copies onto a bus
```

So a reader needs **MoveIt installed** and **Project C checked out beside this
repository**, at `../free-floating-manipulation`. Neither repository
redistributes Franka's model, which is why it is derived at runtime rather than
committed, and that choice is what puts MoveIt on the requirements list.

With those present:

```bash
.venv/bin/python examples/generate_dual_arm_urdf.py   # the model, seven properties
.venv/bin/python examples/dimension_count.py          # what the second arm buys
.venv/bin/python examples/grasp_transition.py         # the mass ratio sweep
.venv/bin/python examples/berthing.py                 # the two regimes
.venv/bin/python examples/derive_mount_spacing.py     # why the mounts are where they are
.venv/bin/python examples/generate_figures.py         # the three figures above
```

`generate_dual_arm_urdf.py` must be run first; everything else loads what it
writes. The rank-loss search inside `dimension_count.py` is the slowest thing
here at about five minutes; the rest are seconds to a minute.

## Structure

```
src/
  dual_arm_model.py     generate the two-arm URDF, and the seven properties
  posture.py            inverse kinematics that respects joint limits
  reference.py          the free-floating scene, and layers A, B and D
  constrained.py        the constrained solve, written twice
  dimension.py          rank and null space of H_bm
  grasp_transition.py   the plant before and after the module joins it
  vendor/               kinematics.py and dynamics.py from Project C,
                        with PROVENANCE.md naming every deviation
examples/               one script per result, each printing what it measured
tests/                  71 tests, each paired with the break it catches
docs/                   the three figures, regenerable from examples/
```

## License

MIT, see [LICENSE](LICENSE).

This covers the code in this repository. The Franka Panda model is not included
here: it is derived at runtime from MoveIt's
`moveit_resources_panda_description`, and the kinematic and inertial values
every result depends on are Franka Emika's, under their own terms. The two
files under `src/vendor/` are copied from `free-floating-manipulation`, which
is MIT and the same author's; `src/vendor/PROVENANCE.md` names the source
commit and every deviation from it.
