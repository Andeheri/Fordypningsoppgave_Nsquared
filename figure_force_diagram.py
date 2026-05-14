"""
figure_force_diagram.py
-----------------------
Publication figure: cable force application on the finger.

The finger is shown at its spring-equilibrium configuration.
Flexion cable forces are shown in red; extension cable forces in blue.
No simulation is performed - this is a purely geometric illustration.

Geometry mirrors adaptive_control/mrac_with_finger_dynamics.py exactly.
"""

import os
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ── Finger geometry (dynamics.py) ──────────────────────────────────────────
l0 = 0.048                       # metacarpal length          [m]
l1, l2, l3 = 0.048, 0.030, 0.024  # PP / MP / DP lengths      [m]

phi1_eq = pi / 6                 # MCP rest angle  ≈ 30°
phi2_eq = pi / 4                 # PIP rest angle  = 45°
phi3_eq = pi / 12                # DIP rest angle  ≈ 15°

# ── Force-attachment parameters (mrac_with_finger_dynamics.py) ─────────────
force_s1, force_s2, force_s3 = l1 * 0.5, l2 * 0.5, l3 * 0.5
r_c = 0.010                      # pulley / ring radius [m]  (same on all links)
aim_frac = 0.5                   # targets at midpoint of the proximal link
d_e = 0.010                      # extension dorsal offset    [m]

# ── Absolute joint angles at equilibrium ───────────────────────────────────
a1 = phi1_eq
a2 = phi1_eq + phi2_eq
a3 = phi1_eq + phi2_eq + phi3_eq

# ── Joint positions (origin = MCP joint) ───────────────────────────────────
O   = np.array([0.0, 0.0])                           # MCP joint
PIP = O   + l1 * np.array([cos(a1), sin(a1)])        # PIP joint (end of PP)
DIP = PIP + l2 * np.array([cos(a2), sin(a2)])        # DIP joint (end of MP)
TIP = DIP + l3 * np.array([cos(a3), sin(a3)])        # fingertip
MC0 = O   - np.array([l0, 0.0])                     # proximal end of metacarpal

# ── Flexion attachment points (90° CCW: offset = r_c * [-sin a, +cos a]) ──
att1_f = (O   + force_s1 * np.array([cos(a1), sin(a1)])
              + r_c      * np.array([-sin(a1),  cos(a1)]))
att2_f = (PIP + force_s2 * np.array([cos(a2), sin(a2)])
              + r_c      * np.array([-sin(a2),  cos(a2)]))
att3_f = (DIP + force_s3 * np.array([cos(a3), sin(a3)])
              + r_c      * np.array([-sin(a3),  cos(a3)]))

# ── Extension attachment points (90° CW: offset = r_c * [+sin a, -cos a]) ─
att1_e = (O   + force_s1 * np.array([cos(a1), sin(a1)])
              + r_c      * np.array([ sin(a1), -cos(a1)]))
att2_e = (PIP + force_s2 * np.array([cos(a2), sin(a2)])
              + r_c      * np.array([ sin(a2), -cos(a2)]))
att3_e = (DIP + force_s3 * np.array([cos(a3), sin(a3)])
              + r_c      * np.array([ sin(a3), -cos(a3)]))

# ── Flexion cable target points ─────────────────────────────────────────────
# Targets lie on the midpoint of the proximal segment, then shifted by d_palm
# in the palmar direction (90° CW from the link axis = [sin a, -cos a]).
d_palm = 0.012   # [m] palmar offset so targets sit clearly behind the finger
target1_f = (np.array([-aim_frac * l0, 0.0])
             + d_palm * np.array([sin(0.0), -cos(0.0)]))           # behind metacarpal
target2_f = (aim_frac * l1 * np.array([cos(a1), sin(a1)])
             + d_palm * np.array([sin(a1), -cos(a1)]))             # behind PP
target3_f = (PIP + aim_frac * l2 * np.array([cos(a2), sin(a2)])
             + d_palm * np.array([sin(a2), -cos(a2)]))             # behind MP

# ── Extension cable target points ───────────────────────────────────────────
target1_e = np.array([0.0, -d_e])                                  # below MCP joint
target2_e = PIP + d_e * np.array([ sin(a1), -cos(a1)])            # dorsal of PIP
target3_e = DIP + d_e * np.array([ sin(a2), -cos(a2)])            # dorsal of DIP

# ── Helpers ─────────────────────────────────────────────────────────────────
MM = 1000.0   # m → mm

def _mm(p):
    return p * MM

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(2)

# ── Visual constants (colours from visualization/visualization_finger_simulation.py) ────
FLEX_C  = "#c0392b"    # deep red   – flexion
EXT_C   = "#1a5276"    # dark blue  – extension
LINK_C  = "#24c6c2"    # phalanx fill    (link_color)
OUTL_C  = "#0f6f6d"    # phalanx outline (outline_color)
JOINT_C = "#2c3e50"    # joint dots      (joint_dot_color)
MC_C    = "#8b7355"    # metacarpal fill (wrist_color)
MC_OC   = "#4a3d2e"    # metacarpal outline (wrist_outline)

LINK_LW  = 18          # link linewidth [pt]
ARR_LEN  = 13.0        # force-arrow display length [mm]
ARR_LW   = 2.8         # force-arrow linewidth [pt]
ATT_MS   = 8           # attachment dot marker size (radius in mm, via scatter s)
JOINT_S  = 80          # joint scatter size
FONT_SM  = 13
FONT_MD  = 15
FONT_LG  = 24

# ── Disc centres (midpoint along each phalanx where the ring sits) ──────────
disc_c1 = O   + force_s1 * np.array([cos(a1), sin(a1)])
disc_c2 = PIP + force_s2 * np.array([cos(a2), sin(a2)])
disc_c3 = DIP + force_s3 * np.array([cos(a3), sin(a3)])

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

# ---------- draw links ----------

def _draw_link(p0, p1, fc, oc, lw=LINK_LW):
    pm0, pm1 = _mm(p0), _mm(p1)
    ax.plot([pm0[0], pm1[0]], [pm0[1], pm1[1]], '-',
            linewidth=lw, solid_capstyle='round', color=fc, alpha=0.92, zorder=2,
            path_effects=[pe.Stroke(linewidth=lw + 5, foreground=oc), pe.Normal()])

_draw_link(MC0, O,   MC_C,  MC_OC)    # metacarpal
_draw_link(O,   PIP, LINK_C, OUTL_C)  # PP
_draw_link(PIP, DIP, LINK_C, OUTL_C)  # MP
_draw_link(DIP, TIP, LINK_C, OUTL_C)  # DP

# ---------- discs (rings on each phalanx) ----------

from matplotlib.patches import Circle
DISC_C    = "#ecf0f1"   # disc fill – light neutral
DISC_EC   = "#7f8c8d"   # disc edge – mid grey
DISC_R_MM = r_c * MM     # disc radius in mm

for dc in [disc_c1, disc_c2, disc_c3]:
    dm = _mm(dc)
    circle = Circle(dm, DISC_R_MM, linewidth=1.5,
                    edgecolor=DISC_EC, facecolor=DISC_C, alpha=0.85, zorder=1)
    ax.add_patch(circle)

# ---------- joint dots ----------

for pt in [O, PIP, DIP, TIP]:
    pm = _mm(pt)
    ax.scatter(pm[0], pm[1], s=JOINT_S, c=JOINT_C,
               edgecolors='white', linewidths=1.8, zorder=5)

# ---------- cable path (dashed) + force arrow + attachment dot ----------

def _draw_force(att, target, color):
    am = _mm(att)
    tm = _mm(target)
    d  = _unit(target - att)

    # thin dashed cable path from attachment to target
    ax.plot([am[0], tm[0]], [am[1], tm[1]], '--',
            color=color, lw=0.9, alpha=0.45, zorder=3, solid_capstyle='round')

    # small X at the target (cable anchor)
    ax.scatter(tm[0], tm[1], s=120, marker='x', c=color,
               linewidths=2.5, zorder=4, alpha=0.85)

    # bold force arrow originating at the attachment point
    end = am + ARR_LEN * d
    ax.annotate('', xy=end, xytext=am,
                arrowprops=dict(
                    arrowstyle='->', color=color,
                    lw=ARR_LW, mutation_scale=20,
                    connectionstyle='arc3,rad=0.0',
                ),
                zorder=7)

    # filled dot at attachment
    ax.scatter(am[0], am[1], s=100, c=color,
               edgecolors='white', linewidths=1.5, zorder=8)


# flexion set
for att, tgt in [(att1_f, target1_f), (att2_f, target2_f), (att3_f, target3_f)]:
    _draw_force(att, tgt, FLEX_C)

# extension set
for att, tgt in [(att1_e, target1_e), (att2_e, target2_e), (att3_e, target3_e)]:
    _draw_force(att, tgt, EXT_C)

# ---------- legend ----------

flex_patch = mpatches.Patch(color=FLEX_C, label='Flexion force')
ext_patch  = mpatches.Patch(color=EXT_C,  label='Extension force')

# proxy artists for disc, attachment dot, and cable target
disc_proxy = mpatches.Patch(facecolor=DISC_C, edgecolor=DISC_EC, linewidth=1.5,
                            label='Pulley disc')
att_proxy = ax.scatter([], [], s=100, c='#555555', edgecolors='white',
                       linewidths=1.5, label='Force attachment point')
tgt_proxy = ax.scatter([], [], s=120, marker='x', c='#555555',
                       linewidths=2.5, label='Cable target point')

ax.legend(
    handles=[flex_patch, ext_patch, disc_proxy, att_proxy, tgt_proxy],
    fontsize=FONT_SM + 0.5,
    loc='upper left',
    framealpha=0.93,
    edgecolor=OUTL_C,
    handlelength=1.6,
)

# ---------- axes formatting ----------

ax.set_aspect('equal')
ax.set_xlabel('x  [mm]', fontsize=FONT_MD)
ax.set_ylabel('y  [mm]', fontsize=FONT_MD)
ax.set_title(
    'Cable force application',
    fontsize=FONT_LG,
    pad=10,
)
ax.grid(True, alpha=0.22, linestyle='--', linewidth=0.7)
ax.tick_params(labelsize=FONT_MD)

# tighten the view around the finger with a small margin
all_pts = np.vstack([_mm(p) for p in [MC0, O, PIP, DIP, TIP,
                                       att1_f, att2_f, att3_f,
                                       att1_e, att2_e, att3_e,
                                       target1_f, target2_f, target3_f,
                                       target1_e, target2_e, target3_e]])
margin = 14.0
ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'figures', 'force_diagram.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved → {out_path}")

thesis_figure_path = r"C:\Users\Anders\OneDrive - NTNU\Fordypningsoppgave - Nsquared\specialization_project\Thesis\Figures\Controller_design"

plt.savefig(os.path.join(thesis_figure_path, 'force_diagram.pdf'), dpi=300, bbox_inches='tight')
