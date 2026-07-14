"""
Oncolytic Virus vs. Cancer Cell Model — V3
Resistance & Combination Therapy (72-hour simulation)

Compares two scenarios side by side:
  1. Virus only — a resistant subpopulation (5% of cells) survives the virus
     and regrows, causing a tumor "bounce-back" (relapse).
  2. Virus + Chemotherapy — a drug bolus at Hour 12 kills both susceptible
     and resistant cells, preventing relapse.

Model (solved with scipy.integrate.solve_ivp):

    dCs/dt = r*Cs*(1 - C/Cmax) - k*V*Cs - kd*D*Cs   (susceptible cancer)
    dCr/dt = r*Cr*(1 - C/Cmax)          - kd*D*Cr   (resistant cancer)
    dV/dt  = p*V*Cs/(1 + V/Vmax) - d*V              (virus replicates in Cs only)
    dD/dt  = -dd*D                                   (drug, bolus at t = 12 h)

C = Cs + Cr (total cancer).  D = 0 in the virus-only scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---- Simulation parameters ----
HOURS = 72
T_EVAL = np.linspace(0, HOURS, 1000)

# ---- Biological constants ----
r = 0.04                # tumor growth rate (per hour)
Cmax = 1.0e7            # carrying capacity (10 million cells)
k = 2.0e-7              # virus infection / killing rate
p = 5.0e-7              # viral replication rate
d = 0.02                # viral clearance rate (per hour)
Vmax = 1.0e8            # viral carrying capacity (prevents unbounded growth)
resistance_frac = 0.05  # 5% of cancer cells are resistant to the virus

# ---- Drug (chemotherapy) parameters ----
DRUG_TIME = 12.0        # hour the drug is administered
D0 = 1.0e5              # bolus drug concentration at injection
kd = 2.0e-6             # drug-mediated killing rate
dd = 0.03               # drug clearance rate (per hour)

# ---- Initial conditions ----
C0 = 1.0e6              # 1 million cancer cells at start
V0 = 1.0e4              # 10,000 viral particles injected
Cs0 = C0 * (1 - resistance_frac)
Cr0 = C0 * resistance_frac

def ode_system(t, y, with_drug):
    Cs, Cr, V, D = y
    Cs = max(Cs, 0)
    Cr = max(Cr, 0)
    V  = max(V,  0)
    D  = max(D,  0)

    C_total = Cs + Cr
    growth_limit = 1.0 - C_total / Cmax

    dCs = r * Cs * growth_limit - k * V * Cs - kd * D * Cs
    dCr = r * Cr * growth_limit              - kd * D * Cr
    dV  = p * V * Cs / (1.0 + V / Vmax) - d * V
    dD  = -dd * D

    return [dCs, dCr, dV, dD]

def simulate(with_drug=False):
    """
    Simulate using solve_ivp with a drug bolus event handled by
    splitting the integration into two segments at DRUG_TIME.
    """
    y0 = [Cs0, Cr0, V0, 0.0]

    if not with_drug:
        sol = solve_ivp(
            ode_system, [0, HOURS], y0,
            args=(False,), t_eval=T_EVAL,
            method="RK45", rtol=1e-8, atol=1e-6,
            dense_output=True
        )
        Cs, Cr, V, D = sol.y
        return Cs, Cr, V, D

    # Phase 1: 0 -> DRUG_TIME
    t1_eval = T_EVAL[T_EVAL <= DRUG_TIME]
    sol1 = solve_ivp(
        ode_system, [0, DRUG_TIME], y0,
        args=(True,), t_eval=t1_eval,
        method="RK45", rtol=1e-8, atol=1e-6
    )

    # Apply bolus at DRUG_TIME
    y_mid = sol1.y[:, -1].copy()
    y_mid[3] += D0  # inject drug

    # Phase 2: DRUG_TIME -> HOURS
    t2_eval = T_EVAL[T_EVAL > DRUG_TIME]
    sol2 = solve_ivp(
        ode_system, [DRUG_TIME, HOURS], y_mid,
        args=(True,), t_eval=t2_eval,
        method="RK45", rtol=1e-8, atol=1e-6
    )

    Cs = np.concatenate([sol1.y[0], sol2.y[0]])
    Cr = np.concatenate([sol1.y[1], sol2.y[1]])
    V  = np.concatenate([sol1.y[2], sol2.y[2]])
    D  = np.concatenate([sol1.y[3], sol2.y[3]])
    return Cs, Cr, V, D

# ---- Run both scenarios ----
Cs_a, Cr_a, V_a, _   = simulate(with_drug=False)
Cs_b, Cr_b, V_b, D_b = simulate(with_drug=True)

C_a = Cs_a + Cr_a
C_b = Cs_b + Cr_b

t_a = T_EVAL
t_b_phase1 = T_EVAL[T_EVAL <= DRUG_TIME]
t_b_phase2 = T_EVAL[T_EVAL > DRUG_TIME]
t_b = np.concatenate([t_b_phase1, t_b_phase2])

# ---- Colors ----
c_cancer = "#d62728"
c_resist = "#ff7f0e"
c_virus  = "#1f77b4"
c_drug   = "#2ca02c"

# ---- Build the two-panel figure ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))

def style_panel(ax, title):
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Cancer cells", color=c_cancer, fontsize=12)
    ax.tick_params(axis="y", labelcolor=c_cancer)
    ax.set_xlim(0, HOURS)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, alpha=0.25)

# --- Left panel: Virus only ---
style_panel(axL, "Virus Only — Resistance Causes Relapse")
axL.plot(t_a, C_a,  color=c_cancer, linewidth=2.4, label="Total cancer (C)")
axL.plot(t_a, Cr_a, color=c_resist, linewidth=2.0, linestyle="--",
         label="Resistant cancer (Cr)")
axV1 = axL.twinx()
axV1.set_ylabel("Viral particles", color=c_virus, fontsize=12)
axV1.tick_params(axis="y", labelcolor=c_virus)
axV1.plot(t_a, V_a, color=c_virus, linewidth=2.0, label="Virus (V)")
axV1.set_ylim(0, V_a.max() * 1.2)

axL.set_ylim(0, C_a.max() * 1.2)

nadir_a = int(np.argmin(C_a))
axL.annotate("Tumor nadir —\nresistant cells regrow",
             xy=(t_a[nadir_a], C_a[nadir_a]),
             xytext=(t_a[nadir_a] + 10, C_a[nadir_a] + C_a.max() * 0.3),
             fontsize=10, color=c_resist, ha="left",
             arrowprops=dict(arrowstyle="->", color=c_resist, lw=1.5))

h1, l1 = axL.get_legend_handles_labels()
h2, l2 = axV1.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)

# --- Right panel: Virus + Chemotherapy ---
style_panel(axR, "Virus + Chemotherapy — Combination Beats Resistance")
axR.plot(t_b, C_b,  color=c_cancer, linewidth=2.4, label="Total cancer (C)")
axR.plot(t_b, Cr_b, color=c_resist, linewidth=2.0, linestyle="--",
         label="Resistant cancer (Cr)")
axV2 = axR.twinx()
axV2.set_ylabel("Viral particles", color=c_virus, fontsize=12)
axV2.tick_params(axis="y", labelcolor=c_virus)
axV2.plot(t_b, V_b, color=c_virus, linewidth=2.0, label="Virus (V)")
axV2.set_ylim(0, V_b.max() * 1.2)

c_max_b = max(C_b.max(), 1)
axR.set_ylim(0, c_max_b * 1.2)

axR.axvline(DRUG_TIME, color=c_drug, linestyle=":", linewidth=2.0,
            label=f"Chemo drug (t = {DRUG_TIME:.0f}h)")
axR.annotate("Chemo drug\nadministered",
             xy=(DRUG_TIME, c_max_b * 0.55),
             xytext=(DRUG_TIME + 6, c_max_b * 0.72),
             fontsize=10, color=c_drug, ha="left",
             arrowprops=dict(arrowstyle="->", color=c_drug, lw=1.5))

h3, l3 = axR.get_legend_handles_labels()
h4, l4 = axV2.get_legend_handles_labels()
axR.legend(h3 + h4, l3 + l4, loc="upper right", fontsize=10)

# ---- Figure-level title ----
fig.suptitle("Oncolytic Virus Therapy: Resistance vs. Combination Treatment",
             fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
plt.savefig("oncolytic_virus_graph.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Summary statistics ----
print("=" * 62)
print("SIMULATION RESULTS — 72-HOUR ONCOLYTIC VIRUS MODEL (V3)")
print("=" * 62)

print("\n[1] Virus Only (with resistance)")
print(f"    Starting cancer cells:  {C_a[0]:,.0f}")
print(f"    Cancer at nadir:        {C_a[nadir_a]:,.0f}  (hour {t_a[nadir_a]:.0f})")
print(f"    Final cancer cells:     {C_a[-1]:,.0f}")
print(f"    Resistant cells at end: {Cr_a[-1]:,.0f}")
print(f"    Peak viral particles:   {V_a.max():,.0f}")

nadir_b = int(np.argmin(C_b))
print("\n[2] Virus + Chemotherapy (drug at Hour 12)")
print(f"    Starting cancer cells:  {C_b[0]:,.0f}")
print(f"    Cancer at nadir:        {C_b[nadir_b]:,.0f}  (hour {t_b[nadir_b]:.0f})")
print(f"    Final cancer cells:     {C_b[-1]:,.0f}")
print(f"    Resistant cells at end: {Cr_b[-1]:,.0f}")
print(f"    Peak viral particles:   {V_b.max():,.0f}")

reduction = (1 - C_b[-1] / C_a[-1]) * 100 if C_a[-1] > 0 else 0
print(f"\n>>> Combination therapy reduced final tumor burden by {reduction:.1f}%")
print(f"    compared to virus alone.")
print("\nGraph saved as 'oncolytic_virus_graph.png'.")
