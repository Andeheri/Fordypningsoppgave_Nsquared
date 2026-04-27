from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.patches import Arc, FancyArrow, Circle
import numpy as np
from numpy import sin, cos
import os


def _move_to_secondary(fig):
    """Move a matplotlib figure window to the secondary monitor (Windows)."""
    try:
        import ctypes
        import ctypes.wintypes

        class _MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong),
                        ("rcMonitor", ctypes.wintypes.RECT),
                        ("rcWork",    ctypes.wintypes.RECT),
                        ("dwFlags",   ctypes.c_ulong)]

        monitors = []
        def _cb(hMon, _hdc, _lprc, _data):
            info = _MONITORINFO()
            info.cbSize = ctypes.sizeof(info)
            ctypes.windll.user32.GetMonitorInfoW(hMon, ctypes.byref(info))
            monitors.append({'left': info.rcMonitor.left,
                             'top':  info.rcMonitor.top,
                             'primary': bool(info.dwFlags & 1)})
            return True

        _PROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong,
                                   ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_double)
        ctypes.windll.user32.EnumDisplayMonitors(None, None, _PROC(_cb), 0)

        target = next((m for m in monitors if not m['primary']), monitors[0] if monitors else None)
        if target is None:
            return
        x, y = target['left'] + 50, target['top'] + 50

        mgr = fig.canvas.manager
        try:
            mgr.window.wm_geometry(f"+{x}+{y}")  # TkAgg
        except AttributeError:
            mgr.window.move(x, y)                 # Qt backends
    except Exception:
        pass


def plot_simulation_angles(t, th1, th2, th3, theta1_0, theta2_0, theta3_0, filepath, should_save):
    fig, ax = plt.subplots(figsize=(9, 5))
    _move_to_secondary(fig)

    ax.plot(t, th1, label=r'$\theta_1$')
    ax.plot(t, th2, label=r'$\theta_2$')
    ax.plot(t, th3, label=r'$\theta_3$')

    # Spring rest angles as dotted horizontal lines
    ax.axhline(theta1_0, color='C0', linestyle=':', linewidth=1.2, label=r'$\theta_{1,0}$')
    ax.axhline(theta2_0, color='C1', linestyle=':', linewidth=1.2, label=r'$\theta_{2,0}$')
    ax.axhline(theta3_0, color='C2', linestyle=':', linewidth=1.2, label=r'$\theta_{3,0}$')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Angle [rad]')
    ax.set_title('Finger joint angles over time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if should_save:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        plt.savefig(filepath, dpi=300)  # Save the plot as a PNG file
    plt.show(block=False)


def animate_finger_simulation(sol, l1, l2, l3, speed=1.0, save_fps=None,
                              link_force_s=None, link_force_mag=None,
                              link_force_r=None, aim_frac=None, l0=None):
    """
    Animate the simulated finger motion.

    Parameters
    ----------
    sol           : ODE solution from solve_ivp
    speed         : playback speed multiplier (default 1.0 = real-time)
    save_fps      : if set, use frame-based timing (required for saving to file).
    link_force_s  : tuple (s1, s2, s3) – attachment distances [m] from each
                    link's proximal joint along the link axis;  0 <= s_i <= l_i.
    link_force_mag: tuple of three callables  F_i(t) -> float [N], one per link.
    link_force_r  : tuple (r1, r2, r3) – circle radii [m] at the midpoints.
    aim_frac      : tuple (aim1, aim2, aim3) – fractional target positions (0-1)
                    along the link below where each force aims.
                    Link 1 aims at the metacarpal (length l0),
                    link 2 at link 1, link 3 at link 2.
    l0            : metacarpal length [m], required when aim_frac is not None.
    """
    save_mode = save_fps is not None
    # ---- Visual style (mirrors visualization_finger_interactive.py) ----
    link_color       = "#24c6c2"
    outline_color    = "#0f6f6d"
    wrist_color      = "#8b7355"
    wrist_outline    = "#4a3d2e"
    angle_color      = "gray"
    link_width       = 20
    outline_offset   = 3
    link_alpha       = 0.8
    joint_dot_size   = 50
    joint_dot_color  = "#2c3e50"
    joint_dot_edge   = "white"
    joint_dot_lw     = 1.5

    # Link lengths in mm
    L0 = l0 * 1000.0 if l0 is not None else 0.0
    # Link lengths in mm
    L1, L2, L3  = l1 * 1000, l2 * 1000, l3 * 1000
    wrist_len   = 0.15 * L1
    ext_len     = 0.25 * L1
    ref_len     = 0.35 * L1
    arc_r1      = 0.25 * L1
    arc_r2      = 0.18 * L1
    arc_r3      = 0.16 * L1
    ann_mult    = 1.5

    base = np.array([0.0, 0.0])

    fig, ax = plt.subplots(figsize=(7, 7))
    _move_to_secondary(fig)
    margin = 1.1 * (L1 + L2 + L3)
    ax.set_xlim(-wrist_len - 5, margin)
    ax.set_ylim(-margin * 0.3, margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [mm]", fontsize=13)
    ax.set_ylabel("y [mm]", fontsize=13)
    ax.set_title("Finger simulation", fontsize=16)
    ax.grid(True, alpha=0.3)

    # ---- Helper draw functions ----
    def _link_infill(p0, p1, color):
        return ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-",
                       linewidth=link_width, solid_capstyle="round",
                       color=color, alpha=link_alpha, zorder=1)[0]

    def _link_rim(p0, p1, color, rim):
        line, = ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-",
                        linewidth=link_width, solid_capstyle="round",
                        color=color, alpha=link_alpha, zorder=4)
        line.set_path_effects([
            pe.Stroke(linewidth=link_width + outline_offset, foreground=rim),
            pe.Normal()
        ])
        return line

    def _draw_ray(origin, angle, length):
        end = origin + length * np.array([cos(angle), sin(angle)])
        return ax.plot([origin[0], end[0]], [origin[1], end[1]],
                       linestyle=":", color=angle_color, zorder=3)[0]

    def _add_arc(center, a1_deg, a2_deg, radius):
        if a2_deg < a1_deg:
            a2_deg += 360
        arc = Arc(center, 2 * radius, 2 * radius,
                  angle=0, theta1=a1_deg, theta2=a2_deg,
                  color=angle_color, zorder=3)
        ax.add_patch(arc)
        return arc

    def _annotate(center, angle1, angle2, radius, label):
        bisector = (angle1 + angle2) / 2
        pos = center + radius * ann_mult * np.array([cos(bisector), sin(bisector)])
        return ax.text(pos[0], pos[1], label, color=angle_color,
                       ha="center", va="center", fontsize=12)

    # ---- Build all artists for the first frame ----
    th1_0, th2_0, th3_0_val = sol.y[0, 0], sol.y[1, 0], sol.y[2, 0]
    a1 = th1_0
    a2 = th1_0 + th2_0
    a3 = th1_0 + th2_0 + th3_0_val

    P_MCP = base + L1 * np.array([cos(a1), sin(a1)])
    P_PIP = P_MCP + L2 * np.array([cos(a2), sin(a2)])
    P_DIP = P_PIP + L3 * np.array([cos(a3), sin(a3)])
    wrist  = base - np.array([wrist_len, 0.0])

    fill_wrist  = _link_infill(wrist, base, wrist_color)
    fill_pp     = _link_infill(base, P_MCP, link_color)
    fill_ip     = _link_infill(P_MCP, P_PIP, link_color)
    fill_dp     = _link_infill(P_PIP, P_DIP, link_color)
    rim_wrist   = _link_rim(wrist, base, wrist_color, wrist_outline)
    rim_pp      = _link_rim(base, P_MCP, link_color, outline_color)
    rim_ip      = _link_rim(P_MCP, P_PIP, link_color, outline_color)
    rim_dp      = _link_rim(P_PIP, P_DIP, link_color, outline_color)

    joint_scatters = [
        ax.scatter(*p, s=joint_dot_size, c=joint_dot_color,
                   edgecolors=joint_dot_edge, linewidths=joint_dot_lw, zorder=5)
        for p in [base, P_MCP, P_PIP, P_DIP]
    ]

    ray_mcp = _draw_ray(P_MCP, a1, ext_len)
    ray_pip = _draw_ray(P_PIP, a2, ext_len)
    ray_ref = _draw_ray(base, 0.0, ref_len)

    arc_mcp = _add_arc(base,  0.0, np.degrees(a1), arc_r1)
    arc_pip = _add_arc(P_MCP, np.degrees(a1), np.degrees(a2), arc_r2)
    arc_dip = _add_arc(P_PIP, np.degrees(a2), np.degrees(a3), arc_r3)

    ann_mcp = _annotate(base,  0.0, a1, arc_r1, r"$\theta_{1}$")
    ann_pip = _annotate(P_MCP, a1,  a2, arc_r2, r"$\theta_{2}$")
    ann_dip = _annotate(P_PIP, a2,  a3, arc_r3, r"$\theta_{3}$")

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                        fontsize=11, va="top")

    # ---- Link force visual setup ----
    _lf_colors = ["crimson", "darkorange", "mediumvioletred"]
    _lf_labels = [r"$F_1$ (link 1)", r"$F_2$ (link 2)", r"$F_3$ (link 3)"]
    _lf_arrow_len = 0.45 * L1   # fixed visual arrow length [mm]
    link_arrow_patches = [None, None, None]   # redrawn every frame
    lf_att_scatters    = []                   # small diamonds at attachment points
    lf_circle_patches  = []                   # circles at link midpoints
    lf_aim_scatters    = []                   # dots on the target link

    def _target_points_mm(a1, a2, a3):
        """Return the three aim-target points in mm (on the link below each force)."""
        MCP_loc = base + L1 * np.array([cos(a1), sin(a1)])
        # target1: along metacarpal (pointing left at angle pi)
        tgt1 = np.array([-aim_frac[0] * L0, 0.0])
        # target2: along link 1
        tgt2 = aim_frac[1] * L1 * np.array([cos(a1), sin(a1)])
        # target3: along link 2, starting from MCP
        tgt3 = MCP_loc + aim_frac[2] * L2 * np.array([cos(a2), sin(a2)])
        return tgt1, tgt2, tgt3

    def _midpoints_mm(a1, a2, a3):
        """Return the three force midpoints (along-link only) in mm."""
        MCP_loc = base + L1 * np.array([cos(a1), sin(a1)])
        PIP_loc = MCP_loc + L2 * np.array([cos(a2), sin(a2)])
        mid1 = base    + link_force_s[0] * 1000.0 * np.array([cos(a1), sin(a1)])
        mid2 = MCP_loc + link_force_s[1] * 1000.0 * np.array([cos(a2), sin(a2)])
        mid3 = PIP_loc + link_force_s[2] * 1000.0 * np.array([cos(a3), sin(a3)])
        return mid1, mid2, mid3

    def _att_points_mm(a1, a2, a3):
        """Return the three force attachment points (midpoint + radial offset) in mm."""
        mid1, mid2, mid3 = _midpoints_mm(a1, a2, a3)
        if link_force_r is not None:
            att1 = mid1 + link_force_r[0] * 1000.0 * np.array([-sin(a1), cos(a1)])
            att2 = mid2 + link_force_r[1] * 1000.0 * np.array([-sin(a2), cos(a2)])
            att3 = mid3 + link_force_r[2] * 1000.0 * np.array([-sin(a3), cos(a3)])
        else:
            att1, att2, att3 = mid1, mid2, mid3
        return att1, att2, att3

    def _force_angles_mm(a1, a2, a3):
        """Return world-frame force angles [rad] by aiming at the target points."""
        att1, att2, att3 = _att_points_mm(a1, a2, a3)
        tgt1, tgt2, tgt3 = _target_points_mm(a1, a2, a3)
        phi1 = np.arctan2(tgt1[1] - att1[1], tgt1[0] - att1[0])
        phi2 = np.arctan2(tgt2[1] - att2[1], tgt2[0] - att2[0])
        phi3 = np.arctan2(tgt3[1] - att3[1], tgt3[0] - att3[0])
        return phi1, phi2, phi3

    def _make_lf_arrow(att_mm, force_angle, mag_val, color):
        """Create a FancyArrow for a link force (returns None if mag is ~zero)."""
        if abs(mag_val) < 1e-12:
            return None
        dx = _lf_arrow_len * cos(force_angle)
        dy = _lf_arrow_len * sin(force_angle)
        hw = _lf_arrow_len * 0.20
        hl = _lf_arrow_len * 0.28
        return FancyArrow(att_mm[0], att_mm[1], dx, dy,
                          width=hw * 0.3, head_width=hw, head_length=hl,
                          color=color, alpha=0.85, zorder=6,
                          length_includes_head=True)

    if link_force_s is not None:
        t0 = sol.t[0]
        state0_vis = sol.y[:, 0]
        atts_i   = _att_points_mm(a1, a2, a3)
        angles_i = _force_angles_mm(a1, a2, a3)
        for idx in range(3):
            patch = _make_lf_arrow(atts_i[idx], angles_i[idx],
                                   link_force_mag[idx](t0, state0_vis),
                                   _lf_colors[idx])
            link_arrow_patches[idx] = patch
            if patch is not None:
                ax.add_patch(patch)
            sc = ax.scatter(*atts_i[idx], s=40, c=_lf_colors[idx], marker='D',
                            edgecolors='black', linewidths=0.8, zorder=7)
            lf_att_scatters.append(sc)

        # Aim-target dots on the link below
        if aim_frac is not None:
            tgts_i = _target_points_mm(a1, a2, a3)
            for tgt, color in zip(tgts_i, _lf_colors):
                sc = ax.scatter(*tgt, s=60, c=color, marker='o',
                                edgecolors='black', linewidths=0.8, zorder=7)
                lf_aim_scatters.append(sc)

        # Draw circles at link midpoints (if radii provided)
        if link_force_r is not None:
            mids_i = _midpoints_mm(a1, a2, a3)
            for idx, (mid, r_m, color) in enumerate(
                    zip(mids_i, link_force_r, _lf_colors)):
                r_mm = r_m * 1000.0
                circ = Circle(mid, r_mm, fill=False, edgecolor=color,
                               linewidth=1.5, linestyle='--', alpha=0.7, zorder=6)
                ax.add_patch(circ)
                lf_circle_patches.append(circ)

        for color, label in zip(_lf_colors, _lf_labels):
            ax.plot([], [], color=color, lw=2, label=label)
        ax.legend(loc="upper right", fontsize=10)

    # Live readout of force values (bottom-left corner)
    force_info_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                              fontsize=8, va="bottom", family="monospace",
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _set_line(line, p0, p1):
        line.set_xdata([p0[0], p1[0]])
        line.set_ydata([p0[1], p1[1]])

    def update(frame):
        import time as _time
        nonlocal arc_mcp, arc_pip, arc_dip

        if save_mode:
            # Frame-based timing: each frame advances by 1/save_fps seconds of real time
            sim_t = min(frame / save_fps * speed, sol.t[-1])
        else:
            # Wall-clock driven: find which simulation index matches elapsed real time
            if update.t_start is None:
                update.t_start = _time.perf_counter()
            elapsed = (_time.perf_counter() - update.t_start) * speed
            sim_t = min(elapsed, sol.t[-1])

        # Clamp to the last frame once the simulation is done
        frame = int(np.searchsorted(sol.t, sim_t))
        frame = min(frame, sol.y.shape[1] - 1)

        th1 = sol.y[0, frame]
        th2 = sol.y[1, frame]
        th3 = sol.y[2, frame]
        a1  = th1
        a2  = th1 + th2
        a3  = th1 + th2 + th3

        MCP = base + L1 * np.array([cos(a1), sin(a1)])
        PIP = MCP  + L2 * np.array([cos(a2), sin(a2)])
        DIP = PIP  + L3 * np.array([cos(a3), sin(a3)])

        for line, p0, p1 in [
            (fill_wrist, wrist, base), (fill_pp, base, MCP),
            (fill_ip, MCP, PIP),       (fill_dp, PIP, DIP),
            (rim_wrist,  wrist, base), (rim_pp,  base, MCP),
            (rim_ip,  MCP, PIP),       (rim_dp,  PIP, DIP),
        ]:
            _set_line(line, p0, p1)

        positions = [base, MCP, PIP, DIP]
        for sc, pos in zip(joint_scatters, positions):
            sc.set_offsets([pos])

        def _ray_end(origin, angle, length):
            e = origin + length * np.array([cos(angle), sin(angle)])
            return [origin[0], e[0]], [origin[1], e[1]]

        ray_mcp.set_xdata(_ray_end(MCP, a1, ext_len)[0])
        ray_mcp.set_ydata(_ray_end(MCP, a1, ext_len)[1])
        ray_pip.set_xdata(_ray_end(PIP, a2, ext_len)[0])
        ray_pip.set_ydata(_ray_end(PIP, a2, ext_len)[1])

        # Update arcs by removing and redrawing
        arc_mcp.remove(); arc_pip.remove(); arc_dip.remove()
        a1d, a2d, a3d = np.degrees(a1), np.degrees(a2), np.degrees(a3)

        def _new_arc(center, start_deg, end_deg, radius):
            s, e = start_deg % 360, end_deg % 360
            if e < s:
                e += 360
            arc = Arc(center, 2*radius, 2*radius, angle=0,
                      theta1=s, theta2=e, color=angle_color, zorder=3)
            ax.add_patch(arc)
            return arc

        arc_mcp = _new_arc(base,       0.0, a1d, arc_r1)
        arc_pip = _new_arc(tuple(MCP), a1d, a2d, arc_r2)
        arc_dip = _new_arc(tuple(PIP), a2d, a3d, arc_r3)

        def _bisector_pos(center, ang1, ang2, r):
            b = (ang1 + ang2) / 2
            return center + r * ann_mult * np.array([cos(b), sin(b)])

        ann_mcp.set_position(_bisector_pos(base, 0.0, a1, arc_r1))
        ann_pip.set_position(_bisector_pos(MCP,  a1,  a2, arc_r2))
        ann_dip.set_position(_bisector_pos(PIP,  a2,  a3, arc_r3))

        time_text.set_text(f"t = {sim_t:.2f} s")

        # Update link force arrows, attachment dots, aim dots, circles, and info text
        if link_force_s is not None:
            att1, att2, att3 = _att_points_mm(a1, a2, a3)
            phi1, phi2, phi3 = _force_angles_mm(a1, a2, a3)
            atts   = [att1, att2, att3]
            phis   = [phi1, phi2, phi3]
            info_lines = []
            for idx in range(3):
                if link_arrow_patches[idx] is not None:
                    link_arrow_patches[idx].remove()
                state_frame = sol.y[:, frame]
                mag_val = link_force_mag[idx](sim_t, state_frame)
                patch = _make_lf_arrow(atts[idx], phis[idx], mag_val, _lf_colors[idx])
                link_arrow_patches[idx] = patch
                if patch is not None:
                    ax.add_patch(patch)
                lf_att_scatters[idx].set_offsets([atts[idx]])
                info_lines.append(f"F{idx+1}={mag_val:6.1f} N")
            force_info_text.set_text("\n".join(info_lines))

            # Update aim-target dots
            if aim_frac is not None:
                tgts = _target_points_mm(a1, a2, a3)
                for sc, tgt in zip(lf_aim_scatters, tgts):
                    sc.set_offsets([tgt])

            # Update circle positions to follow midpoints
            if link_force_r is not None:
                mids = _midpoints_mm(a1, a2, a3)
                for circ, mid in zip(lf_circle_patches, mids):
                    circ.set_center(mid)

        return (fill_wrist, fill_pp, fill_ip, fill_dp,
                rim_wrist, rim_pp, rim_ip, rim_dp,
                ray_mcp, ray_pip, time_text, ann_mcp, ann_pip, ann_dip,
                force_info_text, *joint_scatters, *lf_att_scatters,
                *lf_circle_patches, *lf_aim_scatters)

    update.t_start = None  # will be set on first call (display mode only)

    if save_mode:
        # Exact number of frames needed to cover the full simulation at save_fps
        interval = int(1000 / save_fps)
        n_ticks  = int(np.ceil(sol.t[-1] * save_fps / speed)) + 1
    else:
        # Fire at ~50 fps; each call uses wall time to pick the right frame,
        # so actual playback speed matches real time regardless of render cost.
        # n_frames is set large enough that the animation covers the full duration
        # even on a slow machine.
        interval = 20  # ms (~50 fps target)
        n_ticks  = int(sol.t[-1] * 1000 / interval * (1 / speed)) + 50

    anim = FuncAnimation(fig, update, frames=n_ticks,
                         interval=interval, blit=False, repeat=not save_mode)
    plt.tight_layout()
    return anim
