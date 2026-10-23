"""Quick OPF validation: run 2 h of the optimal-dispatch example and check the OPF controller."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_optimal_dispatch import build_world  # noqa: E402

if __name__ == "__main__":
    import os
    import warnings

    os.chdir(HERE)
    w, opf = build_world(stop_time=7200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        w.simulate(pbar=False)
    for h in opf.history:
        sp = h["setpoints"]
        print(f"t={h['time']:7.0f}s  status={h['solver_status']:.0f}  P2G={sp['Power2Gas.p_mw']:6.2f} MW  "
              f"P2H={sp['Power2Heat.p_mw']:6.2f} MW  grid={sp['grid.p_mw']:7.2f} MW  cost={sp['cost']:9.2f}")
    assert all(h["solver_status"] == 0 for h in opf.history), "OPF did not converge at every step"
    print("OPF controller OK")
