#!/usr/bin/env python3
"""
fast_res.py — the 2° vs 1° resolution experiment of experiments.py (`res`), run in parallel.

tide.py and experiments.py are imported unchanged. Every fit goes through experiments.run_one, so physics,
optimiser, folds and cache keys are identical; finished fits are cached in results/runs and never recomputed.

  python fast_res.py --bench                 # time one 1° factorisation per LU ordering, suggest --workers
  python fast_res.py                         # run all fits in parallel, write results/summary_res.json
  python fast_res.py --solver SYM            # use another ordering, only if --bench shows it is faster and accurate
  python fast_res.py --specs phys,nn:8       # exactly the closures of `experiments.py res`
  python fast_res.py --ref-dir cache/ref/GOT4.10c/GOT4.10c/grids_oceantide_netcdf   # also print the atlas floor
  python experiments.py figures              # afterwards: figs/fig_resolution.pdf
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one thread per process: SuperLU is serial, parallelism is across fits

import argparse, functools, multiprocessing as mp, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import tide as T
import experiments as E

_SPLU = T.spla.splu                         # original scipy routine
SOLVERS = {
    "COLAMD": {},                                                    # scipy default = solver used in the paper
    "MMD_AT_PLUS_A": dict(permc_spec="MMD_AT_PLUS_A"),               # ordering for structurally symmetric matrices
    "SYM": dict(permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.1,   # same ordering + diagonal-preferring pivots
                options=dict(SymmetricMode=True)),
}


def use_solver(name):
    """tide.py calls spla.splu(A); route that call through the chosen SuperLU options."""
    T.spla.splu = functools.partial(_SPLU, **SOLVERS[name])


def _sysctl(key, default):
    try:
        return int(subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).decode())
    except Exception:
        return default


def probe(a, names):
    """Factorise the fine-grid M2 system once per solver; report time, fill, residual, memory and a worker count."""
    lat, lon, H = T.load_bathymetry(a.res_fine)
    m = T.Mesh(lat, lon, H)
    C, om = T.make_closure("phys"), T.omega("M2")
    with torch.no_grad():
        kappa, beta = C.kappa_beta(torch.as_tensor(C.theta0()), m, om)
        z = torch.zeros((m.nlat, m.nlon), dtype=T.CPLX)
        b = -m.pack(*T.residual(m, om, kappa, beta, z, z, z, m.equilibrium("M2")))
    A, b = T.build_system(m, om, kappa, float(beta)), np.ascontiguousarray(b.numpy())
    n_c = len(a.constituents.split(","))
    perf = _sysctl("hw.perflevel0.physicalcpu", max(1, (os.cpu_count() or 2) // 2))
    ram = _sysctl("hw.memsize", 16 * 2 ** 30)
    print(f"[probe] {a.res_fine:g}°: {A.shape[0]} unknowns, {A.nnz} non-zeros; {perf} performance cores, "
          f"{ram / 2 ** 30:.0f} GB RAM")
    out = {}
    for n in names:
        try:
            t0 = time.time(); lu = _SPLU(A, **SOLVERS[n]); tf = time.time() - t0
            t0 = time.time(); x = lu.solve(b); lu.solve(b, trans="H"); ts = (time.time() - t0) / 2
            resid = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
            nnz = int(lu.nnz)
            del lu
            # L-BFGS keeps the first evaluation's graph alive, so up to 2 x n_c factorisations coexist per fit
            mem = 2 * n_c * nnz * 16 * 1.25 + 2 ** 30
            ev = n_c * (tf + 3 * ts)
            out[n] = dict(fact=tf, solve=ts, nnz=nnz, resid=resid, mem=mem, sec_eval=ev,
                          workers=max(1, min(perf, int(0.75 * ram / mem))))
            print(f"[probe] {n:14s} factor {tf:6.2f} s  solve {ts:5.2f} s  fill {nnz / 1e6:6.1f} M  "
                  f"|Ax-b|/|b| {resid:.1e}  ~{ev:.1f} s/eval, ~{75 * ev / 60:.0f} min per NN fit, "
                  f"~{mem / 2 ** 30:.1f} GB per worker -> --workers {out[n]['workers']}")
        except Exception as e:
            print(f"[probe] {n:14s} failed ({type(e).__name__}: {e})")
    return out


def atlas_floor(a, common, rdir):
    """Reference atlas block-averaged to each grid and read through the gauge operator (representativeness floor)."""
    names = a.constituents.split(",")
    ref = T.load_reference_atlas(rdir, names)
    ex, col = T.exclude_ids(), [T.MODEL_SET.index(n) for n in names]
    err, ok, w, ids = {}, None, None, None
    for res in (a.res, a.res_fine):
        ctx = E.get_ctx(a, res, common)
        these = [int(r["id"]) for r in ctx.g]
        assert ids is None or ids == these, "gauge order differs between resolutions"
        ids = these
        gj, gi = np.array([r["j"] for r in ctx.g]), np.array([r["i"] for r in ctx.g])
        Zc = np.stack([T.ref_to_mesh(ctx.m, ref[n])[gj, gi] for n in names], 1)
        good = np.all(np.isfinite(Zc), 1) & np.array([i not in ex for i in ids])
        ok = good if ok is None else ok & good
        err[res] = np.sum(np.abs(np.nan_to_num(Zc) - T.gauge_Z(ctx.g)[:, col]) ** 2, 1)
        w = ctx.w
    print(f"[floor] atlas at model cell, same {int(ok.sum())} gauges: " + ",  ".join(
        f"{res:g}° {100 * np.sqrt(np.sum(w[ok] * e[ok]) / np.sum(w[ok]) / 2):.1f} cm" for res, e in err.items()))


def _work(task):
    a, common, res, spec, fold = task
    torch.set_num_threads(1)
    use_solver(a.solver)
    t0 = time.time()
    rec = E.run_one(E.get_ctx(a, res, common), spec, fold, E.base_cfg(a, res=res))
    return (None if rec is None else float(rec["held_out"])), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=float, default=2.0)                 # defaults below mirror experiments.py
    ap.add_argument("--res-fine", type=float, default=1.0)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--max-gauges", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--constituents", default="M2,S2,K1,O1")
    ap.add_argument("--l2-nn", type=float, default=1e-6)
    ap.add_argument("--l2-phys", type=float, default=1e-3)
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 folds, <=8 L-BFGS iterations")
    ap.add_argument("--specs", default="phys,nn:8,nn:32:2:H+slope+abslat", help="comma-separated closures")
    ap.add_argument("--solver", default="COLAMD", choices=list(SOLVERS))
    ap.add_argument("--workers", type=int, default=None, help="parallel fits (default: from cores and memory)")
    ap.add_argument("--bench", action="store_true", help="only time the LU orderings on the fine grid")
    ap.add_argument("--ref-dir", default=None, help="reference-atlas NetCDF directory: print the floor per grid")
    a = ap.parse_args()
    if a.quick:
        a.iters = min(a.iters, 8)
    for d in (T.CACHE, T.RESULTS, E.RUNS, E.FIELDS, E.FIGS, E.TABS):
        d.mkdir(parents=True, exist_ok=True)

    if a.bench:
        P = {n: p for n, p in probe(a, list(SOLVERS)).items() if p["resid"] < 1e-10}
        if P:
            best = min(P, key=lambda n: P[n]["sec_eval"])
            print(f"[bench] fastest accurate solver: {best}  ->  python fast_res.py --solver {best} "
                  f"--workers {P[best]['workers']}")
        return

    use_solver(a.solver)
    specs = [s for s in a.specs.split(",") if s]
    c2, c1 = E.get_ctx(a, a.res), E.get_ctx(a, a.res_fine)            # builds and caches both gauge sets once
    for res in (a.res, a.res_fine):
        if not (T.CACHE / f"gauges_{a.year}_{res:g}deg_{a.max_gauges}.json").exists():
            sys.exit(f"gauge set at {res:g}° was not cached (download failures); rerun before starting the fits")
    if len(c1.g) >= a.max_gauges:
        print(f"[res] WARNING: the {a.res_fine:g}° gauge set hit --max-gauges; high station ids were dropped")
    common = frozenset({r["id"] for r in c2.g} & {r["id"] for r in c1.g})
    print(f"[res] {len(c2.g)} gauges at {a.res:g}°, {len(c1.g)} at {a.res_fine:g}°, {len(common)} common")

    if a.ref_dir:
        try:
            atlas_floor(a, common, a.ref_dir)
        except Exception as e:
            print(f"[floor] skipped: {type(e).__name__}: {e}")

    if a.workers is None:
        p = probe(a, [a.solver]).get(a.solver)
        a.workers = p["workers"] if p else 4
    tasks = [(a, common, res, s, f) for res in (a.res_fine, a.res) for s in specs for f in E.folds(a)]
    print(f"[res] {len(tasks)} fits, {a.workers} workers, solver {a.solver}")
    t0, failed = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_work, t): t for t in tasks}
        for k, fu in enumerate(as_completed(futs), 1):
            _, _, res, spec, fold = futs[fu]
            try:
                out, sec = fu.result()
                print(f"[done {k:2d}/{len(tasks)}] {res:g}° {spec:26s} fold {fold}: held-out "
                      f"{'n/a' if out is None else f'{out:.1f} cm'}  ({sec:.0f} s; {(time.time() - t0) / 60:.1f} min elapsed)",
                      flush=True)
            except Exception as e:
                failed += 1
                print(f"[FAILED] {res:g}° {spec} fold {fold}: {type(e).__name__}: {e}", flush=True)
    if failed:
        sys.exit(f"{failed} fits failed; finished fits are cached, rerun to retry (try fewer --workers)")

    rows = []                                                         # same summary as experiments.exp_res
    for res in (a.res, a.res_fine):
        ctx = E.get_ctx(a, res, common)
        for r in E.summarise(E.run_cv(ctx, specs, E.base_cfg(a, res=res), E.folds(a)))["rows"]:
            r["res"] = res
            rows.append(r)
            v = r.get("vs_ref")
            print(f"[res] {res:g}° {r['label']:34s} out {r['out_mean']:.1f}±{r['out_sd']:.1f}  pooled {r['pooled']:.1f}  "
                  f"GP {r['pooled_gp']:.1f}  "
                  + (f"vs PHYS {v['diff']:+.1f} [{v['lo']:+.1f},{v['hi']:+.1f}] wins {r['wins_vs_ref']}/{r['n_folds']}  "
                     if v else "")
                  + f"D {r['D']:.2f} TW  shelf {100 * r['shallow']:.0f}%  {r['sec_per_eval']:.2f} s/eval")
    E.save("summary_res.json", dict(rows=rows, n_common=len(common)))
    print(f"[res] wrote {T.RESULTS / 'summary_res.json'}; total wall-clock {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()