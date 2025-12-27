# python libraries
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from IPython.display import HTML, display
from matplotlib.animation import FuncAnimation


def plot_rectangle(center, width, height, ax=None, frame=0.1, **kwargs):
    # make black the default edgecolor
    if not "edgecolor" in kwargs:
        kwargs["edgecolor"] = "black"

    # make transparent the default facecolor
    if not "facecolor" in kwargs:
        kwargs["facecolor"] = "none"

    # get current plot axis if one is not given
    if ax is None:
        ax = plt.gca()

    # get corners
    c2c = np.array([width, height]) / 2
    bottom_left = center - c2c
    top_right = center + c2c

    # plot rectangle
    rect = Rectangle(bottom_left, width, height, **kwargs)
    ax.add_patch(rect)

    # scatter fake corners to update plot limits (bad looking but compact)
    ax.scatter(*bottom_left, s=0)
    ax.scatter(*top_right, s=0)

    # make axis scaling equal
    ax.set_aspect("equal")

    return rect


def animate_footstep_plan_gcs(
        terrain, fl, fr, rl, rr, vertical_limits, horizontal_limits, title="Footstep Plan"
    ):
        """
        Animates the footstep plan with a 2-stage visualization for every step:
        Show Old Positions + Constraint Rectangles (Target Region)
        Show New Positions + Constraint Rectangles (Landed)
        """
        
        # --- Setup Data ---
        feet_traj = [fl, fr, rl, rr] 
        # Total configurations is N. Total transitions (steps) is N-1.
        n_configs = fl.shape[0]
        n_transitions = n_configs - 1
        
        # Mappings
        x_pair_map = {0:2, 1:3, 2:0, 3:1} # Longitudinal
        y_pair_map = {0:1, 1:0, 2:3, 3:2} # Lateral
        
        # Unpack Limits
        v_xmax, v_xmin, v_ymax, v_ymin = vertical_limits
        h_xmax, h_xmin, h_ymax, h_ymin = horizontal_limits
        
        # Box Sizes
        v_width = v_xmax - v_xmin
        v_height = v_ymax - v_ymin
        h_width = h_xmax - h_xmin
        h_height = h_ymax - h_ymin

        # --- Initialize Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        terrain.plot(title=title, ax=ax)

        # Feet Scatters
        feet_colors = ["r", "b", "k", "c"] # FL, FR, RL, RR
        feet_labels = ["FL", "FR", "RL", "RR"]
        scatters = []
        for i in range(4):
            sc = ax.scatter([], [], color=feet_colors[i], s=80, zorder=5, label=feet_labels[i])
            scatters.append(sc)

        # Constraint Rectangles
        rect_long = Rectangle((0,0), h_width, h_height, 
                            fill=False, edgecolor='orange', linestyle='--', linewidth=2, 
                            label='Longitudinal Constraint')
        rect_lat = Rectangle((0,0), v_width, v_height, 
                            fill=False, edgecolor='green', linestyle='--', linewidth=2, 
                            label='Lateral Constraint')
        ax.add_patch(rect_long)
        ax.add_patch(rect_lat)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

        # Helper to calculate box bottom-left corner
        def get_constraint_origin(stance_pos, limits, is_flipped):
            xmax, xmin, ymax, ymin = limits
            sx, sy = stance_pos
            
            if is_flipped:
                # If foot > neighbor: Constraint was -(Swing - Stance)
                # Implies: Swing \in [Stance - Max, Stance - Min]
                return (sx - xmax, sy - ymax)
            else:
                # If foot < neighbor: Constraint was (Swing - Stance)
                # Implies: Swing \in [Stance + Min, Stance + Max]
                return (sx + xmin, sy + ymin)

        def animate(frame_idx):
            # We have 2 frames per transition.
            # frame 0 -> transition 0, phase Old
            # frame 1 -> transition 0, phase New
            # frame 2 -> transition 1, phase Old...
            
            transition_idx = frame_idx // 2
            phase = frame_idx % 2 # 0 = Old Pos, 1 = New Pos
            
            # Define indices for Old (u) and New (v)
            idx_u = transition_idx
            idx_v = transition_idx + 1
            
            # 1. Determine which positions to show
            # If phase 0: Show feet at idx_u
            # If phase 1: Show feet at idx_v
            display_idx = idx_v if phase == 1 else idx_u
            
            current_pos = []
            for foot_idx in range(4):
                pos = feet_traj[foot_idx][display_idx]
                scatters[foot_idx].set_offsets(pos)
                current_pos.append(pos)
                
            # 2. Determine Swing Foot (Based on the transition u->v)
            # The swing foot and boxes depend on the TRANSITION, not the phase.
            # They should be identical for both frames of this pair.
            
            # Find which foot moves between u and v
            swing_idx = -1
            max_dist = 0
            for f in range(4):
                dist = np.linalg.norm(feet_traj[f][idx_v] - feet_traj[f][idx_u])
                if dist > 1e-4:
                    if dist > max_dist:
                        max_dist = dist
                        swing_idx = f
            
            if swing_idx == -1: swing_idx = 0 # Fallback
                
            # 3. Calculate Constraint Boxes
            # The constraints (boxes) are defined by the STANCE feet at the START of the step (idx_u)
            # So we use feet_traj[...][idx_u] to position the boxes.
            
            stance_feet_u = [feet_traj[f][idx_u] for f in range(4)]
            
            # A. Longitudinal Box (Orange)
            neigh_long = x_pair_map[swing_idx]
            flip_long = swing_idx > neigh_long
            xy_long = get_constraint_origin(stance_feet_u[neigh_long], horizontal_limits, flip_long)
            rect_long.set_xy(xy_long)
            
            # B. Lateral Box (Green)
            neigh_lat = y_pair_map[swing_idx]
            flip_lat = swing_idx > neigh_lat
            xy_lat = get_constraint_origin(stance_feet_u[neigh_lat], vertical_limits, flip_lat)
            rect_lat.set_xy(xy_lat)

            return scatters + [rect_long, rect_lat]

        # Run Animation
        # Frames = 2 * number of transitions
        total_frames = n_transitions * 2
        
        ani = FuncAnimation(fig, animate, frames=total_frames, interval=500, blit=True)
        plt.close()
        display(HTML(ani.to_jshtml()))
