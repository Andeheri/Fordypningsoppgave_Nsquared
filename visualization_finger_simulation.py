
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe
from matplotlib.patches import Arc, FancyArrow
import numpy as np
from numpy import sin, cos


def plot_simulation_angles(t, th1, th2, th3, theta1_0, theta2_0, theta3_0, filepath, should_save):
    fig, ax = plt.subplots(figsize=(9, 5))

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
        plt.savefig(filepath, dpi=300)  # Save the plot as a PNG file
    plt.show(block=False)


def animate_finger_simulation(sol, l1, l2, l3, speed=1.0, save_fps=None, force_magnitude=None, force_target=None):
    """
    Animate the simulated finger motion using the same graphical style as the
    visualization scripts.

    Parameters
    ----------
    sol              : ODE solution from solve_ivp
    speed            : playback speed multiplier (default 1.0 = real-time)
    save_fps         : if set, use frame-based timing (required for saving to file);
                       caller is responsible for calling plt.show() afterwards.
    force_magnitude  : if set, draw a red arrow at the fingertip pointing toward
                       `force_target`, scaled visually.
    force_target     : 2-element array-like, target point in metres [m] (default (0, 0)).
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

    # Link lengths converted to mm for display
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

    # ---- Force arrow setup ----
    force_arrow_len = 0.45 * L1  # fixed visual length in mm
    force_arrow_patch = [None]   # mutable container so the nested update() can rebind

    # Convert target from metres to mm for display
    _ft = np.zeros(2) if force_target is None else np.asarray(force_target, dtype=float)
    force_target_mm = _ft * 1000.0

    def _make_force_arrow(tip_mm):
        """Create a FancyArrow from `tip_mm` pointing toward force_target_mm."""
        delta = force_target_mm - tip_mm
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-12:
            return None
        direction = delta / delta_norm
        dx, dy = direction * force_arrow_len
        hw = force_arrow_len * 0.20
        hl = force_arrow_len * 0.28
        return FancyArrow(tip_mm[0], tip_mm[1], dx, dy,
                          width=hw * 0.3, head_width=hw, head_length=hl,
                          color="crimson", alpha=0.85, zorder=6,
                          length_includes_head=True)

    if force_magnitude is not None:
        force_arrow_patch[0] = _make_force_arrow(P_DIP)
        if force_arrow_patch[0] is not None:
            ax.add_patch(force_arrow_patch[0])
        ax.plot([], [], color="crimson", lw=2,
                label=f"F = {force_magnitude:.2g} N → ({_ft[0]:.3g}, {_ft[1]:.3g}) m")
        ax.legend(loc="upper right", fontsize=10)

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

        # Update force arrow
        if force_magnitude is not None:
            if force_arrow_patch[0] is not None:
                force_arrow_patch[0].remove()
            force_arrow_patch[0] = _make_force_arrow(DIP)
            if force_arrow_patch[0] is not None:
                ax.add_patch(force_arrow_patch[0])

        return (fill_wrist, fill_pp, fill_ip, fill_dp,
                rim_wrist, rim_pp, rim_ip, rim_dp,
                ray_mcp, ray_pip, time_text, ann_mcp, ann_pip, ann_dip,
                *joint_scatters)

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