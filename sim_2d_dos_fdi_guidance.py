import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def wrap_angle(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def dos_available(t):
    """Deterministic DoS intervals used for repeatable comparison."""
    blackout = (8.0 <= t <= 13.0) or (22.0 <= t <= 26.0)
    return not blackout


def fdi_bias(sender, t):
    """FDI attacks on the transmitted time-to-go of two compromised vehicles."""
    if not (10.0 <= t <= 33.0):
        return 0.0
    if sender == 1:
        return 12.0 + 2.0 * math.sin(0.55 * t)
    if sender == 3:
        return -9.0 + 0.10 * (t - 10.0)
    return 0.0


def physical_quantities(pos, gamma, speed, target):
    rel = target[None, :] - pos
    r = np.linalg.norm(rel, axis=1)
    lam = np.arctan2(rel[:, 1], rel[:, 0])
    eta = wrap_angle(lam - gamma)
    vc = speed * np.cos(eta)
    tgo = r / np.maximum(vc, 30.0)
    return r, lam, eta, vc, tgo


def run_case(case_name, defense=True, attack=True):
    np.random.seed(2)
    n = 5
    target = np.array([0.0, 0.0])
    speed = np.array([310.0, 292.0, 303.0, 297.0, 315.0])
    pos = np.array(
        [
            [-9200.0, -1700.0],
            [-8500.0, -900.0],
            [-9500.0, 100.0],
            [-8800.0, 950.0],
            [-10100.0, 1750.0],
        ],
        dtype=float,
    )

    gamma = np.arctan2(-pos[:, 1], -pos[:, 0]) + np.deg2rad(
        np.array([8.0, -6.0, 5.0, -8.0, 4.0])
    )
    side = np.sign(pos[:, 1])
    side[side == 0] = 1.0

    dt = 0.02
    t_end = 70.0
    steps = int(t_end / dt)
    hit_radius = 15.0
    amax = 55.0
    k_turn = 1.6
    beta_max = np.deg2rad(40.0)
    k_beta = 0.25

    r0, _, _, _, tgo0 = physical_quantities(pos, gamma, speed, target)
    virtual_tgo0 = float(np.max(tgo0) + 2.5)

    rho = np.ones((n, n))
    last_value = np.tile(tgo0, (n, 1))
    last_time = np.zeros((n, n))
    resid_hist = [[[] for _ in range(n)] for _ in range(n)]

    active = np.ones(n, dtype=bool)
    hit_time = np.full(n, np.nan)
    miss = np.full(n, np.nan)

    hist_t = []
    hist_tgo = []
    hist_cmd = []
    hist_trust_01 = []
    hist_trust_03 = []
    hist_spread = []
    traj = [[] for _ in range(n)]

    for k in range(steps):
        t = k * dt
        r, lam, eta, vc, tgo = physical_quantities(pos, gamma, speed, target)
        virtual_tgo = max(virtual_tgo0 - t, 0.2)

        for i in range(n):
            traj[i].append(pos[i].copy())
            if active[i] and r[i] <= hit_radius:
                active[i] = False
                hit_time[i] = t
                miss[i] = r[i]

        if not np.any(active):
            break

        reports = tgo.copy()
        if attack:
            for j in range(n):
                reports[j] += fdi_bias(j, t)

        reconstructed = np.tile(tgo, (n, 1))
        available = np.ones((n, n), dtype=bool)

        for i in range(n):
            for j in range(n):
                if i == j:
                    available[i, j] = False
                elif attack and not dos_available(t):
                    available[i, j] = False

        if defense:
            norm_resid = np.zeros((n, n))
            phy_score = np.ones((n, n))
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if available[i, j]:
                        elapsed = max(t - last_time[i, j], 0.0)
                        pred = max(last_value[i, j] - elapsed, 0.0)
                        eps = 1.2 + 0.12 * elapsed + 0.018 * elapsed**2
                        res = reports[j] - pred
                        norm_resid[i, j] = res / eps
                        excess = max(abs(res) - eps, 0.0)
                        phy_score[i, j] = math.exp(-1.5 * (excess / eps) ** 2)

            for i in range(n):
                idx = [j for j in range(n) if j != i and available[i, j]]
                if idx:
                    med = float(np.median([norm_resid[i, j] for j in idx]))
                    report_med = float(np.median([reports[j] for j in idx]))
                else:
                    med = 0.0
                    report_med = tgo[i]
                for j in range(n):
                    if i == j:
                        continue
                    if available[i, j]:
                        residual_consistency = math.exp(-0.8 * abs(norm_resid[i, j] - med))
                        report_consistency = math.exp(-0.35 * max(abs(reports[j] - report_med) - 3.0, 0.0))
                        group_score = min(residual_consistency, report_consistency)
                        resid_hist[i][j].append(float(norm_resid[i, j]))
                        resid_hist[i][j] = resid_hist[i][j][-40:]
                        mean_abs = float(np.mean(np.abs(resid_hist[i][j])))
                        kl_like_score = math.exp(-0.55 * max(mean_abs - 1.0, 0.0))
                        rho_bar = phy_score[i, j] * group_score * kl_like_score
                        rho[i, j] += dt * 4.0 * (rho_bar - rho[i, j])
                        rho[i, j] = float(np.clip(rho[i, j], 0.0, 1.0))

                        if rho[i, j] > 0.85:
                            last_value[i, j] = reports[j]
                            last_time[i, j] = t

                        reconstructed[i, j] = rho[i, j] * reports[j] + (1.0 - rho[i, j]) * virtual_tgo
                    else:
                        reconstructed[i, j] = virtual_tgo
        else:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if available[i, j]:
                        reconstructed[i, j] = reports[j]
                    else:
                        reconstructed[i, j] = tgo[i]

        cmd = np.zeros(n)
        for i in range(n):
            if not active[i]:
                continue
            neigh = [j for j in range(n) if j != i]
            ref = float(np.mean([reconstructed[i, j] for j in neigh]))
            delay_need = max(ref - tgo[i], 0.0)
            terminal_gate = float(np.clip((tgo[i] - 4.0) / 9.0, 0.0, 1.0))
            beta = side[i] * beta_max * math.tanh(k_beta * delay_need) * terminal_gate
            gamma_des = lam[i] + beta
            gdot = k_turn * wrap_angle(gamma_des - gamma[i])
            gdot = float(np.clip(gdot, -amax / speed[i], amax / speed[i]))
            gamma[i] += gdot * dt
            cmd[i] = speed[i] * gdot
            pos[i, 0] += speed[i] * math.cos(gamma[i]) * dt
            pos[i, 1] += speed[i] * math.sin(gamma[i]) * dt

        alive_tgo = tgo[active] if np.any(active) else tgo
        hist_t.append(t)
        hist_tgo.append(tgo.copy())
        hist_cmd.append(cmd.copy())
        hist_trust_01.append(rho[0, 1])
        hist_trust_03.append(rho[0, 3])
        hist_spread.append(float(np.max(alive_tgo) - np.min(alive_tgo)))

    for i in range(n):
        if np.isnan(hit_time[i]):
            r, _, _, _, _ = physical_quantities(pos, gamma, speed, target)
            hit_time[i] = t_end
            miss[i] = r[i]

    result = {
        "case": case_name,
        "hit_time_mean": float(np.mean(hit_time)),
        "hit_time_std": float(np.std(hit_time)),
        "hit_time_range": float(np.max(hit_time) - np.min(hit_time)),
        "max_miss": float(np.max(miss)),
        "mean_miss": float(np.mean(miss)),
        "min_trust_link_0_1": float(np.min(hist_trust_01)) if hist_trust_01 else 1.0,
        "min_trust_link_0_3": float(np.min(hist_trust_03)) if hist_trust_03 else 1.0,
    }

    return {
        "name": case_name,
        "result": result,
        "t": np.array(hist_t),
        "tgo": np.array(hist_tgo),
        "cmd": np.array(hist_cmd),
        "spread": np.array(hist_spread),
        "trust01": np.array(hist_trust_01),
        "trust03": np.array(hist_trust_03),
        "traj": [np.array(x) for x in traj],
        "target": target,
    }


def make_plots(cases):
    colors = {
        "No attack": "#2f6f4e",
        "Hybrid attack without defense": "#b23a48",
        "Hybrid attack with proposed defense": "#2d5f9a",
    }

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))

    ax = axes[0, 0]
    for case in cases:
        for i, tr in enumerate(case["traj"]):
            if i == 0:
                ax.plot(tr[:, 0] / 1000, tr[:, 1] / 1000, color=colors[case["name"]], lw=1.6, label=case["name"])
            else:
                ax.plot(tr[:, 0] / 1000, tr[:, 1] / 1000, color=colors[case["name"]], lw=0.9, alpha=0.65)
    ax.scatter([0], [0], s=36, marker="x", color="black", label="Target")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Planar trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for case in cases:
        ax.plot(case["t"], case["spread"], color=colors[case["name"]], lw=1.8, label=case["name"])
    ax.axvspan(8, 13, color="#999999", alpha=0.15)
    ax.axvspan(22, 26, color="#999999", alpha=0.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tgo spread (s)")
    ax.set_title("Time-to-go synchronization error")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    defended = next(c for c in cases if c["name"] == "Hybrid attack with proposed defense")
    ax.plot(defended["t"], defended["trust01"], color="#7a5195", lw=1.8, label="trust: receiver 0, sender 1")
    ax.plot(defended["t"], defended["trust03"], color="#ef5675", lw=1.8, label="trust: receiver 0, sender 3")
    ax.axvspan(10, 33, color="#d95f02", alpha=0.12, label="FDI interval")
    ax.axvspan(8, 13, color="#999999", alpha=0.15, label="DoS interval")
    ax.axvspan(22, 26, color="#999999", alpha=0.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trust")
    ax.set_title("Trust attenuation on attacked links")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    labels = [case["name"].replace("Hybrid attack ", "Hybrid\nattack\n") for case in cases]
    ranges = [case["result"]["hit_time_range"] for case in cases]
    bars = ax.bar(labels, ranges, color=[colors[c["name"]] for c in cases], width=0.62)
    for bar, val in zip(bars, ranges):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Impact-time range (s)")
    ax.set_title("Terminal synchronization")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim_2d_dos_fdi_guidance_results.png", bbox_inches="tight")
    plt.close(fig)


def main():
    cases = [
        run_case("No attack", defense=True, attack=False),
        run_case("Hybrid attack without defense", defense=False, attack=True),
        run_case("Hybrid attack with proposed defense", defense=True, attack=True),
    ]
    make_plots(cases)
    rows = [case["result"] for case in cases]
    headers = list(rows[0].keys())
    csv_lines = [",".join(headers)]
    for row in rows:
        csv_lines.append(",".join(str(row[h]) for h in headers))
    (OUT_DIR / "sim_2d_dos_fdi_guidance_metrics.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    widths = {h: max(len(h), *(len(f"{row[h]:.4f}") if isinstance(row[h], float) else len(str(row[h])) for row in rows)) for h in headers}
    print(" ".join(h.ljust(widths[h]) for h in headers))
    for row in rows:
        values = []
        for h in headers:
            val = row[h]
            values.append((f"{val:.4f}" if isinstance(val, float) else str(val)).ljust(widths[h]))
        print(" ".join(values))
    print(f"\nSaved: {OUT_DIR / 'sim_2d_dos_fdi_guidance_results.png'}")
    print(f"Saved: {OUT_DIR / 'sim_2d_dos_fdi_guidance_metrics.csv'}")


if __name__ == "__main__":
    main()
