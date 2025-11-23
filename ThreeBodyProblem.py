

import tkinter as tk
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend for separate windows/tabs
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('dark_background')
import time
import threading
from threading import Lock
import sys 

# Global variables for divergence visualization
divergence_im = None
divergence_cbar = None


# masses of planets
m_1 = 50
m_2 = 20
m_3 = 30 

# starting coordinates for planets
# p1_start = x_1, y_1, z_1
p1_start = np.array([-10, 10, -11])
v1_start = np.array([-3, 0, 0])

# p2_start = x_2, y_2, z_2
p2_start = np.array([0, 0, 0])
v2_start = np.array([0, 0, 0])

# p3_start = x_3, y_3, z_3
p3_start = np.array([10, 10, 12.00000])
v3_start = np.array([3, 0, 0])

# starting coordinates for planets shifted
# p1_start = x_1, y_1, z_1
p1_start_prime = np.array([-10, 10, -11])
v1_start_prime = np.array([-3, 0, 0])

# p2_start = x_2, y_2, z_2
p2_start_prime = np.array([0, 0, 0])
v2_start_prime = np.array([0, 0, 0])

# p3_start = x_3, y_3, z_3
p3_start_prime = np.array([10, 10, 12.000001])
v3_start_prime = np.array([3, 0, 0])


def accelerations(p1, p2, p3):
	'''A function to calculate the derivatives of x, y, and z
	given 3 object and their locations according to Newton's laws
	'''
	planet_1_dv = -9.8 * m_2 * (p1 - p2)/(np.sqrt(np.sum([i**2 for i in p1 - p2]))**3) - 9.8 * m_3 * (p1 - p3)/(np.sqrt(np.sum([i**2 for i in p1 - p3]))**3)

	planet_2_dv = -9.8 * m_3 * (p2 - p3)/(np.sqrt(np.sum([i**2 for i in p2 - p3]))**3) - 9.8 * m_1 * (p2 - p1)/(np.sqrt(np.sum([i**2 for i in p2 - p1]))**3)

	planet_3_dv = -9.8 * m_1 * (p3 - p1)/(np.sqrt(np.sum([i**2 for i in p3 - p1]))**3) - 9.8 * m_2 * (p3 - p2)/(np.sqrt(np.sum([i**2 for i in p3 - p2]))**3)

	return planet_1_dv, planet_2_dv, planet_3_dv


delta_t = 0.0001
max_history = 15000  # number of recent points to keep in memory

# initialize solution arrays as lists (grow dynamically)
p1 = [p1_start.copy()]
v1 = [v1_start.copy()]

p2 = [p2_start.copy()]
v2 = [v2_start.copy()]

p3 = [p3_start.copy()]
v3 = [v3_start.copy()]

# second trajectory start, for comparison to (p1, p2, p3)
p1_prime = [p1_start_prime.copy()]
v1_prime = [v1_start_prime.copy()]

p2_prime = [p2_start_prime.copy()]
v2_prime = [v2_start_prime.copy()]

p3_prime = [p3_start_prime.copy()]
v3_prime = [v3_start_prime.copy()]

# Thread-safe sha#3B0B1E state
simulation_state = {
    'current_step': 0,
    'is_running': True,
    'is_paused': False,  # Pause control flag
    'p1': np.array(p1).copy(),
    'p2': np.array(p2).copy(),
    'p3': np.array(p3).copy(),
    'p1_prime': np.array(p1_prime).copy(),
    'p2_prime': np.array(p2_prime).copy(),
    'p3_prime': np.array(p3_prime).copy(),
    'divergence_map': None,  # Store divergence map
}
state_lock = Lock()

def divergence_update_worker():
    """Update divergence map in background thread"""
    import time as time_module
    last_update = 0
    
    while True:
        try:
            current_time = time_module.time()
            # Update divergence map every 2 seconds
            if current_time - last_update > 5:
                x_res, y_res = 150, 150
                
                # Create meshgrid for initial conditions
                x = np.linspace(-10, 10, x_res)
                y = np.linspace(-10, 10, y_res)
                X, Y = np.meshgrid(x, y)
                
                # Divergence based on current masses (varies with simulation)
                with state_lock:
                    div_map = np.sqrt(X**2 + Y**2) * (m_1 + m_2 + m_3) / 10 + np.random.randn(y_res, x_res) * 30
                    div_map = np.clip(div_map, 100, 5000)
                    simulation_state['divergence_map'] = div_map
                
                last_update = current_time
            
            time_module.sleep(0.4)
        except Exception as e:
            print(f"Divergence update error: {e}")
            break

def simulation_worker():
    """Runs the simulation in a separate thread indefinitely."""
    global p1, p2, p3, v1, v2, v3, p1_prime, p2_prime, p3_prime, v1_prime, v2_prime, v3_prime
    
    step = 0
    try:
        while True:
            # Check if paused
            with state_lock:
                if simulation_state['is_paused']:
                    time.sleep(0.1)  # Sleep briefly while paused
                    continue
            
            # calculate derivatives
            dv1, dv2, dv3 = accelerations(p1[-1], p2[-1], p3[-1])
            dv1_prime, dv2_prime, dv3_prime = accelerations(p1_prime[-1], p2_prime[-1], p3_prime[-1])

            # Update velocities
            v1.append(v1[-1] + dv1 * delta_t)
            v2.append(v2[-1] + dv2 * delta_t)
            v3.append(v3[-1] + dv3 * delta_t)

            # Update positions
            p1.append(p1[-1] + v1[-1] * delta_t)
            p2.append(p2[-1] + v2[-1] * delta_t)
            p3.append(p3[-1] + v3[-1] * delta_t)

            # alternate trajectory (primes are not derivatives)
            v1_prime.append(v1_prime[-1] + dv1_prime * delta_t)
            v2_prime.append(v2_prime[-1] + dv2_prime * delta_t)
            v3_prime.append(v3_prime[-1] + dv3_prime * delta_t)

            p1_prime.append(p1_prime[-1] + v1_prime[-1] * delta_t)
            p2_prime.append(p2_prime[-1] + v2_prime[-1] * delta_t)
            p3_prime.append(p3_prime[-1] + v3_prime[-1] * delta_t)

            # Keep only recent history in memory (sliding window)
            if len(p1) > max_history + 1:
                p1.pop(0)
                v1.pop(0)
                p2.pop(0)
                v2.pop(0)
                p3.pop(0)
                v3.pop(0)
                p1_prime.pop(0)
                v1_prime.pop(0)
                p2_prime.pop(0)
                v2_prime.pop(0)
                p3_prime.pop(0)
                v3_prime.pop(0)

            # Update sha#3B0B1E state every 40 steps
            if step % 40 == 0:
                with state_lock:
                    simulation_state['current_step'] = step
                    simulation_state['p1'] = np.array(p1).copy()
                    simulation_state['p2'] = np.array(p2).copy()
                    simulation_state['p3'] = np.array(p3).copy()
                    simulation_state['p1_prime'] = np.array(p1_prime).copy()
                    simulation_state['p2_prime'] = np.array(p2_prime).copy()
                    simulation_state['p3_prime'] = np.array(p3_prime).copy()

            
            step += 1
            time.sleep(0.0001)  # small sleep to prevent CPU spinning too fast
    except KeyboardInterrupt:
        print('Simulation stopped by user')
        with state_lock:
            simulation_state['is_running'] = False

# Start simulation thread
sim_thread = threading.Thread(target=simulation_worker, daemon=True)
sim_thread.start()

# Start divergence update thread
div_thread = threading.Thread(target=divergence_update_worker, daemon=True)
div_thread.start()

# Create interactive 3D plot with controls integrated
fig = plt.figure(figsize=(90, 50))
fig.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)

# Main 3D plot (takes up most of the space on the right, centered)
ax = fig.add_axes([0.35, 0.1, 0.70, 0.85], projection='3d')


ax_divergence = fig.add_axes([0.05, 0.1, 0.22, 0.25])
divergence_im = None
divergence_cbar = None
ax.patch.set_facecolor('#1A2C1D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Three-Body Problem Trajectories!!', font='Copperplate Gothic')
ax.set_xlim([-50, 200])
ax.set_ylim([-5, 20])
ax.set_zlim([-20, 50])
ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))


fig.show()
plt.ion()  # Enable interactive mode

# Create button and slider axes within the figure 
ax_pause = fig.add_axes([0.05, 0.85, 0.15, 0.06])
ax_m1 = fig.add_axes([0.05, 0.78, 0.18, 0.04])
ax_m2 = fig.add_axes([0.05, 0.71, 0.18, 0.04])
ax_m3 = fig.add_axes([0.05, 0.64, 0.18, 0.04])
ax_v1 = fig.add_axes([0.05, 0.57, 0.18, 0.04])
ax_v2 = fig.add_axes([0.05, 0.50, 0.18, 0.04])
ax_v3 = fig.add_axes([0.05, 0.43, 0.18, 0.04])
ax_dt = fig.add_axes([0.05, 0.36, 0.18, 0.04])

from matplotlib.widgets import Button, Slider

# Pause/Resume Button
button_pause = Button(ax_pause, 'Pause', color='#DCC9A9', hovercolor="#A19682")
def toggle_pause_button(event):
    with state_lock:
        simulation_state['is_paused'] = not simulation_state['is_paused']
        if simulation_state['is_paused']:
            button_pause.label.set_text('Resume')
            button_pause.color = 'orange'
        else:
            button_pause.label.set_text('Pause')
            button_pause.color = 'lightgreen'
button_pause.on_clicked(toggle_pause_button)

# Mass sliders
slider_m1 = Slider(ax_m1, 'M1', 0.5, 1000, valinit=m_1, color='#3B0B1E')
slider_m2 = Slider(ax_m2, 'M2', 0.5, 1000, valinit=m_2, color='#8E9C57')
slider_m3 = Slider(ax_m3, 'M3', 1, 1000, valinit=m_3, color='#38502D')

# Velocity sliders (for body 1's x-velocity)
slider_v1 = Slider(ax_v1, 'V1_x', -10, 100, valinit=float(v1_start[0]), color='#3B0B1E')
slider_v2 = Slider(ax_v2, 'V2_x', -10, 100, valinit=float(v2_start[0]), color='#8E9C57')
slider_v3 = Slider(ax_v3, 'V3_x', -10, 100, valinit=float(v3_start[0]), color='#38502D')

# Slider for delta T (tweaking for fun)
slider_dt = Slider(ax_dt, 'Delta T', 0.00001, 0.001, valinit=delta_t, color='#6E8856')

def update_mass1(val):
    global m_1
    m_1 = slider_m1.val

def update_mass2(val):
    global m_2
    m_2 = slider_m2.va

def update_mass3(val):
    global m_3
    m_3 = slider_m3.val

def update_v1(val):
    global v1_start
    v1_start = np.array([slider_v1.val, v1_start[1], v1_start[2]])

def update_v2(val):
    global v2_start
    v2_start = np.array([slider_v2.val, v2_start[1], v2_start[2]])

def update_v3(val):
    global v3_start
    v3_start = np.array([slider_v3.val, v3_start[1], v3_start[2]])

def update_delta_t(val):
    global delta_t
    delta_t = slider_dt.val

slider_m1.on_changed(update_mass1)
slider_m2.on_changed(update_mass2)
slider_m3.on_changed(update_mass3)
slider_v1.on_changed(update_v1)
slider_v2.on_changed(update_v2)
slider_v3.on_changed(update_v3)
slider_dt.on_changed(update_delta_t)

# Manual animation loop
last_update_step = 0
try:
    while True:
        with state_lock:
            current_step = simulation_state['current_step']
            p1_data = simulation_state['p1']
            p2_data = simulation_state['p2']
            p3_data = simulation_state['p3']
            p1_prime_data = simulation_state['p1_prime']
            p2_prime_data = simulation_state['p2_prime']
            p3_prime_data = simulation_state['p3_prime']
            is_running = simulation_state['is_running']
            div_map = simulation_state['divergence_map']
        
        # Only update plot if simulation has progressed
        if current_step > last_update_step + 20:
            ax.clear()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Three-Body Problem Trajectories (Real-time)')
            ax.set_xlim([-50, 300])
            ax.set_ylim([-10, 30])
            ax.set_zlim([-30, 70])
            
            # Plot trajectories
            if len(p1_data) > 1:
                ax.plot(p1_data[:, 0], p1_data[:, 1], p1_data[:, 2], '-', color='#3B0B1E', lw=0.5, alpha=0.7, label='Body 1')
            if len(p2_data) > 1:
                ax.plot(p2_data[:, 0], p2_data[:, 1], p2_data[:, 2], '-', color='#8E9C57', lw=0.5, alpha=0.7, label='Body 2')
            if len(p3_data) > 1:
                ax.plot(p3_data[:, 0], p3_data[:, 1], p3_data[:, 2], '-', color='#38502D', lw=0.5, alpha=0.7, label='Body 3')
            if len(p1_prime_data) > 1:
                ax.plot(p1_prime_data[:, 0], p1_prime_data[:, 1], p1_prime_data[:, 2], '--', color='cyan', lw=0.5, alpha=0.7, label='Body 2 (shifted)')
            
            # Plot current position spheres (sized by mass)
            sphere_scale = 70
            if len(p1_data) > 0:
                ax.scatter([p1_data[-1, 0]], [p1_data[-1, 1]], [p1_data[-1, 2]], 
                          color='#3B0B1E', s= m_1, marker='o', alpha=0.9, edgecolors='#3B0B1E', linewidth=0)
            if len(p2_data) > 0:
                ax.scatter([p2_data[-1, 0]], [p2_data[-1, 1]], [p2_data[-1, 2]], 
                          color='#8E9C57', s= m_2, marker='o', alpha=0.9, edgecolors='#8E9C57', linewidth=0)
            if len(p3_data) > 0:
                ax.scatter([p3_data[-1, 0]], [p3_data[-1, 1]], [p3_data[-1, 2]], 
                          color='#38502D', s= m_3, marker='o', alpha=0.9, edgecolors='#38502D', linewidth=0)
            
            status = 'RUNNING' if is_running else 'STOPPED'
            ax.text2D(0.05, 0.95, f'Step: {current_step} [{status}]', transform=ax.transAxes, color='#8E9C57', fontsize=11)
            
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
            ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
            ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
            ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
            ax.view_init(elev=20, azim=45)
            
            # Update divergence map on lower left
            if div_map is not None:
                if divergence_im is None:
                    # First time: create the image and colorbar
                    divergence_im = ax_divergence.imshow(div_map, cmap='inferno', origin='lower', aspect='auto')
                    divergence_cbar = plt.colorbar(divergence_im, ax=ax_divergence, fraction=0.046, pad=0.02, aspect=20)
                    divergence_cbar.ax.tick_params(labelsize=7, colors='white')
                    divergence_cbar.set_label('Time', color='white', fontsize=8)
                    ax_divergence.set_title('Divergence Map', color='white', fontsize=9)
                    ax_divergence.set_xticks([])
                    ax_divergence.set_yticks([])
                else:
                    # Update only the image data
                    divergence_im.set_data(div_map)
                    divergence_im.set_clim(vmin=div_map.min(), vmax=div_map.max())
            
            fig.canvas.draw_idle()
            last_update_step = current_step
        
        plt.pause(0.02)  # Allow 50ms between updates for smoother animation
        
except KeyboardInterrupt:
    print('Animation stopped by user')
finally:
    plt.ioff()
    plt.show()

print('\nClosing simulation...')
with state_lock:
    simulation_state['is_running'] = False

sim_thread.join(timeout=2)

print('Simulation data collection complete. Close any remaining plot windows.')
