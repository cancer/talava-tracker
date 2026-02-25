#!/usr/bin/env python3
"""Anatomical consistency checker for inference_server logs.

Parses FPS lines from inference_server stdout/log and checks that
knee/foot positions are anatomically consistent with hip.

Usage:
    python3 scripts/check_anatomy.py <log_file>
    python3 scripts/check_anatomy.py logs/inference_20260224_164824.log
    tail -500 /path/to/stdout.output | python3 scripts/check_anatomy.py -

Checks performed per frame:
  1. Limbs (knee/foot) must be below hip (dy < 0)
  2. Limb distance from hip must be < 1.5m
  3. Foot must be below knee (if both present on same side)
  4. Knee/foot X should roughly follow hip X (|dx| < 0.5m)
"""

import re
import sys


def parse_fps_line(line):
    """Extract tracker positions from an FPS line."""
    parts = re.findall(r"(\w+)\(([-\d.]+),([-\d.]+),([-\d.]+)\)", line)
    d = {}
    for name, x, y, z in parts:
        d[name] = (float(x), float(y), float(z))
    return d


def check_frame(trackers):
    """Check anatomical consistency. Returns list of issue strings."""
    if "hip" not in trackers:
        return []

    hx, hy, hz = trackers["hip"]
    issues = []

    limb_names = ["L_foot", "R_foot", "L_knee", "R_knee"]
    for name in limb_names:
        if name not in trackers:
            continue
        px, py, pz = trackers[name]
        dx, dy, dz = px - hx, py - hy, pz - hz
        dist = (dx**2 + dy**2 + dz**2) ** 0.5

        # Check 1: limbs must be below hip
        if dy >= 0:
            issues.append(f"{name} ABOVE hip (dy={dy:+.3f})")

        # Check 2: distance from hip
        if dist > 1.5:
            issues.append(f"{name} TOO FAR from hip (d={dist:.3f})")

        # Check 4: X offset from hip
        if abs(dx) > 0.5:
            issues.append(f"{name} X drift (dx={dx:+.3f})")

    # Check 3: foot below knee (same side)
    for side in ["L", "R"]:
        foot_key = f"{side}_foot"
        knee_key = f"{side}_knee"
        if foot_key in trackers and knee_key in trackers:
            foot_y = trackers[foot_key][1]
            knee_y = trackers[knee_key][1]
            if foot_y >= knee_y:
                issues.append(
                    f"{foot_key} ABOVE {knee_key} "
                    f"(foot_y={foot_y:.3f} >= knee_y={knee_y:.3f})"
                )

    return issues


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file_or_->", file=sys.stderr)
        sys.exit(1)

    src = sys.stdin if sys.argv[1] == "-" else open(sys.argv[1])

    total = 0
    bad = 0
    all_issues = []

    for line in src:
        if "FPS:" not in line or "hip(" not in line:
            continue

        trackers = parse_fps_line(line)
        issues = check_frame(trackers)
        total += 1

        if issues:
            bad += 1
            all_issues.append((total, trackers["hip"], issues))

    if src is not sys.stdin:
        src.close()

    # Report
    if total == 0:
        print("No FPS lines with hip data found.")
        sys.exit(1)

    print(f"Checked {total} frames, {bad} with issues ({bad/total*100:.1f}%)")
    print()

    if not all_issues:
        print("ALL OK - no anatomical anomalies detected.")
    else:
        for frame_num, hip, issues in all_issues:
            print(
                f"Frame {frame_num}: hip({hip[0]:.2f},{hip[1]:.2f},{hip[2]:.2f})"
            )
            for issue in issues:
                print(f"  - {issue}")
        print()
        if bad / total > 0.1:
            print(f"WARNING: {bad/total*100:.1f}% of frames have issues.")
        else:
            print(f"Minor: {bad} frames with issues ({bad/total*100:.1f}%).")

    sys.exit(1 if bad > 0 else 0)


if __name__ == "__main__":
    main()
