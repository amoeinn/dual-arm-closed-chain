# Vendored from free-floating-manipulation

Source: `github.com/amoeinn/free-floating-manipulation`
Commit: `343e2152bb6330c857a573f13f13486c68f4a408`, 2026-09-13
Files:  `src/kinematics.py`, `src/dynamics.py`

Copied rather than depended on, for the standard this portfolio already
applies: a reader must be able to obtain the repository and run the results.
Project C rejected OpenSplat, the INRIA reference and gsplat on that standard
and it applies to our own code too. A submodule or an unpublished shared
package would both fail it.

The cost is stated rather than discovered: **a bug fixed here does not reach
Project C.** Backporting is a decision, and this file is where it is recorded.

## Intended deviations from the source

Listed as they are made. Anything not listed is a copy.

1. `dynamics.py`, `FloatingBaseModel._arm_joint_frames`: walks one chain and
   is documented as "shared by every link's Jacobian since every link is
   distal to all of them". False for two arms. Generalised to merge joint
   frames across all chains.

   The merge carries an assertion that two chains reporting the same joint
   agree about its world frame. **That assertion is unreachable on the
   two-arm servicer** and Phase 7 found it by trying to mutate it. Both arms
   hang off the bus through fixed mounts, so the left chain covers joints 1
   to 7 and the right covers 13 to 19 with nothing in common, and the branch
   never executes. It had never fired because it could not, which is not the
   same as passing.

   It is kept rather than deleted, because the generalisation is to N chains
   and not to two arms specifically, and chains that share joints are an
   ordinary topology: two end effectors on one arm, or arms on a common
   turret. It is now tested on that topology, wrist and elbow on the left
   arm, where the chains share joints 1 to 4 and the merge has real work to
   do. Both the point and the axis halves of the disagreement are caught.
2. `dynamics.py`, the end-effector API: `manipulator_jacobian`,
   `base_jacobian` and `generalized_jacobian` are singular in the source and
   become per-arm here.
3. `dynamics.py`: the import of `kinematics` is repointed from
   `src.kinematics` to `src.vendor.kinematics`. Path only.
4. `kinematics.py`: **no change at all.** `ForwardKinematics.__init__` already
   takes the end effector link, so two arms is two instances. Verified: the
   vendored file's sha256 matches `git show 343e2152:src/kinematics.py`
   exactly.

## On PyBullet's calculateJacobian, and a correction

Neither vendored file calls `calculateJacobian`. Both compute their Jacobians
analytically from the URDF, which is most of why they are worth vendoring.
Project C uses the call only in its verification scripts, comparing its
analytic `J_m` against it on a fixed base and agreeing to 1.7e-15.

An earlier note in this project said `calculateJacobian` had been found
unusable and was banned here. **That was wrong and is corrected.** The call
returns the Jacobian of the link FRAME origin when given a `localPosition` of
`[0, 0, 0]`, and checked against a central difference of that point it agrees
to 2.491e-05, which is the finite-difference floor at a 1e-3 step in Bullet's
single precision. It reads 1.239e-02 against the link's centre of mass for the
same reason a ruler disagrees with a thermometer: the centre of mass is a
different point, 31.6 mm away on `panda_hand`. Both of this project's
"failures" were that comparison, made twice.

So there is no divergence to backport on this account, and Project C's
verification of `J_m` stands. Phase 1's results are unaffected, because the
code there now derives everything from positions and from the mass matrix and
does not depend on which conclusion was right. What was wrong was the reason
recorded, not a number.

The general lesson is the one this phase keeps teaching and it is worth
keeping here where the next person vendoring something will read it: before
concluding that a library is wrong, establish which point, frame and
convention its answer is expressed in. Five of Phase 1's seven defects were
that, and so was the accusation against the library.
