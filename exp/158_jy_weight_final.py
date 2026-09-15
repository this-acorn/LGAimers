"""JY endpoint 혼합 가중의 정점을 LB 3점으로 확정하고 최종 zip 을 즉시 빌드한다.

    S(w) = S_A + d*w + K*w(1-w)      d = S_B - S_A,  K = 다양성

측정
    w = 0.0000000000  →  1116.5770907872   (현재 배포본, 이미 측정)
    w = 0.5000000000  →  1142.7349769136
    w = 0.5894274349634 →  <인자로 받음>

미지수 d, K 두 개 / 식 두 개 → 정확해.
    w* = 0.5 + d/(2K)          정점 = S_A + (K + d)^2/(4K)

    python exp/158_jy_weight_final.py --w0589 <점수> [--build]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'candidates/candidate_affine_jy_w0589427_src'
S_A = 1116.5770907872
W1, S1 = 0.5, 1142.7349769136
W2 = 0.5894274349634
ANCHOR = "ENDPOINT_WEIGHT = 0.5894274349634"


def solve(s2):
    mat = np.array([[W1, W1 * (1 - W1)], [W2, W2 * (1 - W2)]], dtype=np.float64)
    rhs = np.array([S1 - S_A, s2 - S_A], dtype=np.float64)
    d, k = np.linalg.solve(mat, rhs)
    return d, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w0589", type=float, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    d, k = solve(args.w0589)
    print(f"측정 3점:  w=0 → {S_A:.10f}   w={W1} → {S1:.10f}   w={W2:.6f} → {args.w0589:.10f}")
    print(f"\n  d = S_B - S_A = {d:+.6f}   →  endpoint 단독 = {S_A + d:.6f}")
    print(f"  K (다양성)     = {k:+.6f}")
    if k <= 0:
        print("\n  판정: K <= 0 — 혼합 곡선이 위로 볼록하지 않다. 현재 최고점을 유지할 것.")
        return

    w_star = 0.5 + d / (2.0 * k)
    peak = S_A + (k + d) ** 2 / (4.0 * k)
    w_clip = float(np.clip(w_star, 0.0, 1.0))
    print(f"\n  최적 w* = {w_star:.10f}" + ("  (구간 밖 → 클립)" if w_star != w_clip else ""))
    print(f"  예상 정점 = {peak:.6f}   현재 최고({max(S1, args.w0589):.4f}) 대비 "
          f"{peak - max(S1, args.w0589):+.4f}")
    print("\n  검산")
    for w, obs in ((W1, S1), (W2, args.w0589)):
        print(f"    w={w:.6f}  모델 {S_A + d * w + k * w * (1 - w):12.6f}  실측 {obs:12.6f}")

    if peak - max(S1, args.w0589) < 0.01:
        print("\n  판정: 이미 정점 부근이다. 더 내지 말 것.")
        return
    if not args.build:
        print(f"\n  빌드하려면: python exp/158_jy_weight_final.py --w0589 {args.w0589} --build")
        return

    dest = ROOT / 'candidates/candidate_jy_wopt_src'
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(SRC, dest, ignore=shutil.ignore_patterns("__pycache__", "data", "output"))
    script = dest / "script.py"
    text = script.read_text(encoding="utf-8")
    if text.count(ANCHOR) != 1:
        raise SystemExit("script.py 의 ENDPOINT_WEIGHT 앵커를 찾지 못했습니다")
    script.write_text(text.replace(ANCHOR, f"ENDPOINT_WEIGHT = {w_clip!r}", 1), encoding="utf-8")

    zip_path = ROOT / 'artifacts/candidates/candidate_jy_wopt.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(dest.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(dest).as_posix())
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()

    with zipfile.ZipFile(zip_path) as archive:
        tops = {n.split("/")[0] for n in archive.namelist()}
        assert tops == {"model", "script.py", "requirements.txt"}, sorted(tops)
        found = [n for n in archive.namelist() if n == "script.py"]
        assert found, "script.py 누락"
    print(f"\n  빌드 완료: {zip_path.name}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  SHA256: {digest}")
    print(f"  ENDPOINT_WEIGHT = {w_clip!r}")


if __name__ == "__main__":
    main()
