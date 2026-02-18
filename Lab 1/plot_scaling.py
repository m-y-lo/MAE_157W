"""
Generate fs vs U and St vs Re scaling plots from the Strouhal table data.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Data from Table (ROI 3, laser/optimized config) ──────────────────────────
# Cylinder
cyl_U     = np.array([0.363, 0.369, 0.404])
cyl_dU    = np.array([0.036, 0.031, 0.045])
cyl_fs    = np.array([6.745, 7.808, 8.984])
cyl_Re    = np.array([237, 241, 264])
cyl_St    = np.array([0.219, 0.249, 0.262])
cyl_dSt   = np.array([0.032, 0.041, 0.041])

# Triangle
tri_U     = np.array([0.363, 0.369, 0.404])
tri_dU    = np.array([0.036, 0.031, 0.045])
tri_fs    = np.array([7.519, 7.559, 6.736])
tri_Re    = np.array([237, 241, 264])
tri_St    = np.array([0.244, 0.241, 0.197])
tri_dSt   = np.array([0.060, 0.051, 0.050])

# Airfoil
air_U     = np.array([0.363, 0.369, 0.404])
air_dU    = np.array([0.036, 0.031, 0.045])
air_fs    = np.array([6.293, 7.177, 8.035])
air_Re    = np.array([237, 241, 264])
air_St    = np.array([0.205, 0.229, 0.235])
air_dSt   = np.array([0.057, 0.050, 0.044])

# ── Plot styling ─────────────────────────────────────────────────────────────
markers = {'Cylinder': 'o', 'Triangle': 's', 'Airfoil': '^'}
colors  = {'Cylinder': '#1f77b4', 'Triangle': '#ff7f0e', 'Airfoil': '#2ca02c'}

# ── Figure 1: fs vs U ───────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 5))

for name, U, dU, fs, mk, cl in [
    ('Cylinder', cyl_U, cyl_dU, cyl_fs, markers['Cylinder'], colors['Cylinder']),
    ('Triangle', tri_U, tri_dU, tri_fs, markers['Triangle'], colors['Triangle']),
    ('Airfoil',  air_U, air_dU, air_fs, markers['Airfoil'],  colors['Airfoil']),
]:
    ax1.errorbar(U, fs, xerr=dU, fmt=mk, color=cl, markersize=8,
                 capsize=4, linewidth=1.5, label=name)
    # Linear fit for trend line
    coeffs = np.polyfit(U, fs, 1)
    U_fit = np.linspace(0.34, 0.43, 50)
    ax1.plot(U_fit, np.polyval(coeffs, U_fit), '--', color=cl, alpha=0.5,
             linewidth=1)

ax1.set_xlabel(r'Free-stream velocity $U$ (m/s)', fontsize=12)
ax1.set_ylabel(r'Shedding frequency $f_s$ (Hz)', fontsize=12)
ax1.set_title(r'Vortex Shedding Frequency vs. Free-Stream Velocity', fontsize=13,
              fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.32, 0.44])
fig1.tight_layout()
fig1.savefig('fs_vs_U.png', dpi=200, bbox_inches='tight')
print('Saved fs_vs_U.png')

# ── Figure 2: St vs Re ──────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))

for name, Re, St, dSt, mk, cl in [
    ('Cylinder', cyl_Re, cyl_St, cyl_dSt, markers['Cylinder'], colors['Cylinder']),
    ('Triangle', tri_Re, tri_St, tri_dSt, markers['Triangle'], colors['Triangle']),
    ('Airfoil',  air_Re, air_St, air_dSt, markers['Airfoil'],  colors['Airfoil']),
]:
    ax2.errorbar(Re, St, yerr=dSt, fmt=mk, color=cl, markersize=8,
                 capsize=4, linewidth=1.5, label=name)

# Expected St band
ax2.axhspan(0.18, 0.22, color='gray', alpha=0.15, label=r'Expected $St \approx 0.18$–$0.22$')
ax2.axhline(y=0.20, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax2.set_xlabel(r'Reynolds number $Re$', fontsize=12)
ax2.set_ylabel(r'Strouhal number $St$', fontsize=12)
ax2.set_title(r'Strouhal Number vs. Reynolds Number', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.05, 0.40])
ax2.set_xlim([220, 280])
fig2.tight_layout()
fig2.savefig('St_vs_Re.png', dpi=200, bbox_inches='tight')
print('Saved St_vs_Re.png')

print('Done.')
