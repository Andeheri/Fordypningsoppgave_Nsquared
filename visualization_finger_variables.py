
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib.patheffects as pe

# ===== CONFIGURATION =====
# Output
image_folder = "figures"
figure_size = (8, 8)
dpi = 300

# Joint angles
theta_MCP = np.pi / 6  # 30 degrees
theta_PIP = np.pi / 4  # 45 degrees
theta_DIP = np.pi / 3  # 60 degrees

# Link lengths (mm)
l_PP = 45.0  # Proximal phalanx length
l_IP = 30.0  # Intermediate phalanx length
l_DP = 20.0  # Distal phalanx length
wrist_length = 15.0  # Wrist extension backward

# Colors
link_color = "#24c6c2"       # Turquoise
outline_color = "#0f6f6d"    # Dark turquoise
wrist_color = "#8b7355"      # Brown
wrist_outline = "#4a3d2e"    # Dark brown
angle_color = "gray"

# Link appearance
link_width = 20
outline_width_offset = 3  # How much wider the outline is
link_alpha = 0.8

# Font sizes
theta_fontsize = 14
axis_label_fontsize = 14
title_fontsize = 20
tick_fontsize = 12

# Angle annotation parameters
arc_radius_mcp = 0.25 * l_PP
arc_radius_pip = 0.18 * l_PP
arc_radius_dip = 0.16 * l_PP
annotation_distance_multiplier = 1.5  # How far from arc center to place text
# Individual annotation offsets
annotation_mcp_x_offset = 0.0
annotation_mcp_y_offset = -1.0
annotation_pip_x_offset = 0.0
annotation_pip_y_offset = 0.0
annotation_dip_x_offset = 0.0
annotation_dip_y_offset = 0.0

# Extension rays
extension_length = 0.25 * l_PP
reference_line_length = 0.35 * l_PP

# Joint markers
joint_dot_size = 50
joint_dot_color = "#2c3e50"
joint_dot_edge_color = "white"
joint_dot_edge_width = 1.5

# Link length annotation bars
length_bar_offset = 0.0
length_bar_tick = 1.5
length_bar_inset = 0.0
length_label_offset = 4.0
length_bar_color = angle_color
length_bar_fontsize = 14
show_length_lines = False
show_length_end_ticks = False

# Plot limits
xlim = (-15, 60)
ylim = (-5, 70)
# ===== END CONFIGURATION =====

# Visualization
plt.figure(figsize=figure_size)

# Additional visualization aligned with joint angles
base = np.array([0.0, 0.0])
a1 = theta_MCP
a2 = theta_MCP + theta_PIP
a3 = theta_MCP + theta_PIP + theta_DIP

P_MCP = base + np.array([l_PP * cos(a1), l_PP * sin(a1)])
P_PIP = P_MCP + np.array([l_IP * cos(a2), l_IP * sin(a2)])
P_DIP = P_PIP + np.array([l_DP * cos(a3), l_DP * sin(a3)])

def draw_ray(origin, angle, length, **kwargs):
    end = origin + length * np.array([cos(angle), sin(angle)])
    plt.plot([origin[0], end[0]], [origin[1], end[1]], **kwargs)

def add_angle_arc(center, angle1, angle2, radius, **kwargs):
    a1 = np.degrees(angle1) % 360
    a2 = np.degrees(angle2) % 360
    if a2 < a1:
        a2 += 360
    arc = Arc(center, 2 * radius, 2 * radius, angle=0, theta1=a1, theta2=a2, **kwargs)
    plt.gca().add_patch(arc)

def annotate_angle(center, angle1, angle2, radius, label, x_offset=0.0, y_offset=0.0):
    bisector = (angle1 + angle2) / 2
    r = radius * annotation_distance_multiplier
    pos = center + r * np.array([cos(bisector), sin(bisector)])
    pos[0] += x_offset
    pos[1] += y_offset
    plt.text(pos[0], pos[1], label, color=angle_color, ha="center", va="center", fontsize=theta_fontsize)

def draw_length_annotation(p0, p1, label, offset=0.0, inset=0.0, label_offset=0.0,
                           draw_line=False, draw_ticks=False):
    v = p1 - p0
    norm = np.linalg.norm(v)
    if norm == 0:
        return
    v_hat = v / norm
    n_hat = np.array([-v_hat[1], v_hat[0]])
    p0_off = p0 + n_hat * offset + v_hat * inset
    p1_off = p1 + n_hat * offset - v_hat * inset
    if draw_line:
        plt.plot([p0_off[0], p1_off[0]], [p0_off[1], p1_off[1]],
                 color=length_bar_color, linewidth=1.5, zorder=6)
        if draw_ticks:
            t0a = p0_off - n_hat * length_bar_tick
            t0b = p0_off + n_hat * length_bar_tick
            t1a = p1_off - n_hat * length_bar_tick
            t1b = p1_off + n_hat * length_bar_tick
            plt.plot([t0a[0], t0b[0]], [t0a[1], t0b[1]], color=length_bar_color, linewidth=1.5, zorder=6)
            plt.plot([t1a[0], t1b[0]], [t1a[1], t1b[1]], color=length_bar_color, linewidth=1.5, zorder=6)
    mid = (p0_off + p1_off) / 2 + n_hat * label_offset
    plt.text(mid[0], mid[1], label, color=length_bar_color,
             ha="center", va="bottom", fontsize=length_bar_fontsize)

def plot_link_infill(p0, p1, label=None, color=None):
    fill_color = color if color else link_color
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "-",
             linewidth=link_width, solid_capstyle="round", solid_joinstyle="round",
             color=fill_color, alpha=link_alpha, label=label, zorder=1)

def plot_link_rim(p0, p1, color=None, rim_color=None):
    fill_color = color if color else link_color
    rim_col = rim_color if rim_color else outline_color
    line, = plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "-",
                     linewidth=link_width, solid_capstyle="round", solid_joinstyle="round",
                     color=fill_color, alpha=link_alpha, zorder=4)
    line.set_path_effects([
        pe.Stroke(linewidth=link_width + outline_width_offset, foreground=rim_col),
        pe.Normal()
    ])

# Wrist (extending outside plot area)
wrist_start = base - np.array([wrist_length, 0.0])

plot_link_infill(wrist_start, base, label="Wrist", color=wrist_color)
plot_link_infill(base, P_MCP, label="Proximal")
plot_link_infill(P_MCP, P_PIP, label="Intermediate")
plot_link_infill(P_PIP, P_DIP, label="Distal")

plot_link_rim(wrist_start, base, color=wrist_color, rim_color=wrist_outline)
plot_link_rim(base, P_MCP)
plot_link_rim(P_MCP, P_PIP)
plot_link_rim(P_PIP, P_DIP)

# Joint and endpoint markers (drawn on top)
joint_positions = [base, P_MCP, P_PIP, P_DIP]
for pos in joint_positions:
    plt.scatter(pos[0], pos[1], s=joint_dot_size, c=joint_dot_color,
                edgecolors=joint_dot_edge_color, linewidths=joint_dot_edge_width, zorder=5)

# Dotted extension rays past joints (only along incoming link)
draw_ray(P_MCP, a1, extension_length, linestyle=":", color=angle_color)
draw_ray(P_PIP, a2, extension_length, linestyle=":", color=angle_color)

# Reference x-axis dotted line from origin and MCP angle arc
draw_ray(base, 0.0, reference_line_length, linestyle=":", color=angle_color)
add_angle_arc(base, 0.0, a1, radius=arc_radius_mcp, color=angle_color)

# Angle arcs between links
add_angle_arc(P_MCP, a1, a2, radius=arc_radius_pip, color=angle_color)
add_angle_arc(P_PIP, a2, a3, radius=arc_radius_dip, color=angle_color)

# Angle annotations (bisector between dotted line and next link)
annotate_angle(base, 0.0, a1, arc_radius_mcp, r"$\theta_{MCP}$", annotation_mcp_x_offset, annotation_mcp_y_offset)
annotate_angle(P_MCP, a1, a2, arc_radius_pip, r"$\theta_{PIP}$", annotation_pip_x_offset, annotation_pip_y_offset)
annotate_angle(P_PIP, a2, a3, arc_radius_dip, r"$\theta_{DIP}$", annotation_dip_x_offset, annotation_dip_y_offset)

# Link length annotations
draw_length_annotation(base, P_MCP, r"$\ell_{PP}$", offset=length_bar_offset, inset=length_bar_inset,
                       label_offset=length_label_offset, draw_line=show_length_lines,
                       draw_ticks=show_length_end_ticks)
draw_length_annotation(P_MCP, P_PIP, r"$\ell_{IP}$", offset=length_bar_offset, inset=length_bar_inset,
                       label_offset=length_label_offset, draw_line=show_length_lines,
                       draw_ticks=show_length_end_ticks)
draw_length_annotation(P_PIP, P_DIP, r"$\ell_{DP}$", offset=length_bar_offset, inset=length_bar_inset,
                       label_offset=length_label_offset, draw_line=show_length_lines,
                       draw_ticks=show_length_end_ticks)
plt.axis("equal")
plt.xlim(xlim)
plt.ylim(ylim)
plt.xlabel("x [mm]", fontsize=axis_label_fontsize)
plt.ylabel("y [mm]", fontsize=axis_label_fontsize)
plt.title("Finger Visualization", fontsize=title_fontsize)
plt.tick_params(axis='both', labelsize=tick_fontsize)
plt.grid(True)
plt.savefig(f"{image_folder}/finger_variables_visualization.png", dpi=dpi, bbox_inches="tight")

