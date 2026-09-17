"""Write the dual-arm servicer URDF and check the five properties on it.

  python examples/generate_dual_arm_urdf.py

Nothing downstream in this project may depend on the model until this reports
all five passing. It is the critical path: no dynamics, no reference and no
test can be checked before the model exists.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dual_arm_model import BUS_MASS, DEFAULT_ARMS, verify, write

ROOT = Path(__file__).resolve().parent.parent
# Project C generates this by expanding the MoveIt xacro; it is a derived
# artifact there too, and neither repository commits Franka's model.
SOURCE = (ROOT.parent / "free-floating-manipulation" / "ws" / "build"
          / "panda_identified.urdf")
DESTINATION = ROOT / "build" / "dual_arm_servicer.urdf"


def main() -> int:
    if not SOURCE.exists():
        print(f"source model missing: {SOURCE}\n"
              f"Project C generates it from the MoveIt panda xacro; run its "
              f"freeflight.identified_urdf() first.", file=sys.stderr)
        return 1

    path = write(SOURCE, DESTINATION)
    print(f"wrote {path.relative_to(ROOT)}")
    print(f"  from {SOURCE}")
    print(f"  arms {[arm.prefix for arm in DEFAULT_ARMS]}, bus {BUS_MASS:.1f} kg\n")

    results = verify(path, SOURCE, DEFAULT_ARMS, BUS_MASS)
    width = max(len(name) for name in results)
    failed = 0
    for name, (ok, detail) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed} of {len(results)} properties failed; nothing "
              f"downstream may use this model", file=sys.stderr)
        return 1
    print(f"all {len(results)} properties hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
