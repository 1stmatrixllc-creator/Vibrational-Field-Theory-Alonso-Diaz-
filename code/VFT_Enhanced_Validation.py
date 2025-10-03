import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import ScaledTranslation
from PIL import Image
import io

# ====================================================================
# FUNCTIONS FROM vft_enhanced.py (INLINED TO AVOID IMPORT ERROR)
# ====================================================================

def run_enhanced_simulation(Nx, Ny, T, snap_frames, field_type, damping, initial_amplitude):
    """
    Simulates wave propagation in a vibrational field.
    Returns simulation snapshots, phase-locked velocities (PLV),
    frequency field, and metadata.
    """
    print(f"Running simulation with parameters: Nx={Nx}, Ny={Ny}, T={T}, frames={snap_frames}")
    
    # Create frequency field based on type
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 0.5, Ny)
    X, Y = np.meshgrid(x, y)

    if field_type == 'gradient':
        f_field = np.ones((Ny, Nx)) * 10.0
        f_field[:, :Nx // 3] = 5.0
        f_field[:, Nx // 3:2 * Nx // 3] = 10.0
        f_field[:, 2 * Nx // 3:] = 20.0
        f_field = gaussian_filter(f_field, sigma=3)
    else:
        f_field = np.ones((Ny, Nx)) * 10.0
    
    v_field = 1.0 / f_field
    
    # Simulation parameters
    dx = 1.0 / Nx
    dt = 0.0005
    steps = int(T / dt)
    
    wave_field = np.zeros((Ny, Nx), dtype=np.complex128)
    source_x, source_y = Nx // 6, Ny // 2
    pulse_width = 5
    for i in range(-pulse_width, pulse_width + 1):
        for j in range(-pulse_width, pulse_width + 1):
            if 0 <= source_y + i < Ny and 0 <= source_x + j < Nx:
                wave_field[source_y + i, source_x + j] = initial_amplitude
    
    prev_field = wave_field.copy()
    snapshots = []
    
    snap_interval = max(1, steps // snap_frames)
    
    for step in range(steps):
        if step % snap_interval == 0:
            snapshots.append(np.abs(wave_field.copy()))
        
        laplacian = np.zeros_like(wave_field)
        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                laplacian[i, j] = (wave_field[i + 1, j] + wave_field[i - 1, j] + 
                                  wave_field[i, j + 1] + wave_field[i, j - 1] - 
                                  4 * wave_field[i, j]) / (dx * dx)
        
        next_field = (2 * wave_field - prev_field + 
                     dt * dt * v_field * v_field * laplacian - 
                     damping * dt * (wave_field - prev_field))
        
        prev_field = wave_field.copy()
        wave_field = next_field.copy()
    
    meta = {
        'Nx': Nx, 'Ny': Ny, 'T': T, 'snap_frames': snap_frames, 
        'field_type': field_type, 'damping': damping, 
        'initial_amplitude': initial_amplitude
    }
    
    return snapshots, None, f_field, meta

def configure_publication_plot():
    """
    Applies global Matplotlib settings for publication-quality figures.
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    rcParams['axes.labelsize'] = 9
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['axes.titlesize'] = 11
    rcParams['legend.fontsize'] = 9
    rcParams['font.size'] = 9

def add_panel_label(ax, label, x_offset=0.01, y_offset=0.98):
    """Adds a panel label (A, B, C) to a subplot."""
    ax.text(x_offset, y_offset, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

# ====================================================================

# Create output directory
if not os.path.exists('validation_results'):
    os.makedirs('validation_results')

# Apply publication formatting
configure_publication_plot()

print("Running simulation with better visualization parameters...")
# Run simulation with parameters that produce visible waves
Nx, Ny = 64, 64
T = 1.0  # Shorter simulation time
snap_frames = 50
vft_snaps, vft_plv, f_field, meta = run_enhanced_simulation(
    Nx=Nx, Ny=Ny, T=T, snap_frames=snap_frames, 
    field_type='gradient', 
    damping=0.001,  # Much less damping to preserve wave structure
    initial_amplitude=2.0  # Stronger initial wave
)

print("Analyzing wave data...")
# Convert vft_snaps to a NumPy array for advanced slicing
vft_snaps = np.array(vft_snaps)

# Check wave amplitudes across time
amplitudes = []
for i in range(snap_frames):
    if i >= len(vft_snaps):
        continue
    amp = np.abs(vft_snaps[i])
    amplitudes.append({
        'min': np.min(amp),
        'max': np.max(amp),
        'mean': np.mean(amp),
        'std': np.std(amp)
    })
    print(f"Frame {i}: min={amplitudes[i]['min']:.4f}, max={amplitudes[i]['max']:.4f}, " +
          f"mean={amplitudes[i]['mean']:.4f}, std={amplitudes[i]['std']:.4f}")

# =============================================
# FIX 1: WAVE PROPAGATION ANALYSIS FIGURE
# =============================================
print("\nFix 1: Creating improved Wave Propagation Analysis figure...")

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

# Plot frequency field
im0 = axes[0, 0].imshow(f_field, origin='lower', cmap='viridis')
axes[0, 0].set_title('Frequency Field f(x)')
axes[0, 0].set_xlabel('x (grid units)')
axes[0, 0].set_ylabel('y (grid units)')
plt.colorbar(im0, ax=axes[0, 0], label='Frequency (Hz)')
add_panel_label(axes[0, 0], 'A')

# Calculate wave energy over time
energy = np.sum(np.abs(vft_snaps)**2, axis=(1, 2))

# Plot energy over time
time_points = np.linspace(0, T, snap_frames)
axes[0, 1].plot(time_points, energy, '-', color='#009E73', linewidth=1.5)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Total Energy (a.u.)')
axes[0, 1].set_title('Wave Energy Over Time')
axes[0, 1].grid(True, alpha=0.3, linestyle=':')
add_panel_label(axes[0, 1], 'B')

# Region signals - with clearer regions
region1 = np.mean(np.abs(vft_snaps[:, 10:20, 10:20]), axis=(1, 2))
region2 = np.mean(np.abs(vft_snaps[:, 40:50, 40:50]), axis=(1, 2))
axes[1, 0].plot(time_points, region1, '-', color='#0072B2', linewidth=1.5, label='Low Frequency')
axes[1, 0].plot(time_points, region2, '-', color='#D55E00', linewidth=1.5, label='High Frequency')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Field Amplitude (a.u.)')
axes[1, 0].set_title('Wave Amplitude in Different Regions')
axes[1, 0].legend(frameon=False)
axes[1, 0].grid(True, alpha=0.3, linestyle=':')
add_panel_label(axes[1, 0], 'C')

# Plot final wave amplitude
mid_frame = snap_frames // 2
amplitude = np.abs(vft_snaps[mid_frame])
p5 = np.percentile(amplitude, 5)
p95 = np.percentile(amplitude, 95)
amplitude_scaled = np.clip(amplitude, p5, p95)

im2 = axes[1, 1].imshow(amplitude_scaled, origin='lower', cmap='inferno')
axes[1, 1].set_title('Wave Amplitude (Mid-simulation)')
axes[1, 1].set_xlabel('x (grid units)')
axes[1, 1].set_ylabel('y (grid units)')
plt.colorbar(im2, ax=axes[1, 1], label='Amplitude (a.u.)')
add_panel_label(axes[1, 1], 'D')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Wave Propagation Analysis in VFT', fontsize=12)

# Save in PDF and TIFF formats
plt.savefig('validation_results/Fig2_wave_propagation_fixed.pdf', dpi=600, bbox_inches='tight')
plt.savefig('validation_results/Fig2_wave_propagation_fixed.tiff', dpi=600, bbox_inches='tight')
plt.close(fig)

# =============================================
# FIX 2: WAVE PROPAGATION TIMELINE FIGURE
# =============================================
print("\nFix 2: Creating improved Wave Propagation Timeline figure...")

# Double-column figure width for timeline (7.2 inches width)
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5))
frames = [0, snap_frames//8, snap_frames//4, snap_frames//2, 3*snap_frames//4]
frame_times = np.linspace(0, T, snap_frames)

all_amplitudes = np.concatenate([np.abs(vft_snaps[f]).flatten() for f in frames])
vmin = 0
vmax = np.percentile(all_amplitudes, 95)

for i, frame in enumerate(frames):
    amplitude = np.abs(vft_snaps[frame])
    if np.any(np.isnan(amplitude)) or np.any(np.isinf(amplitude)):
        print(f"Warning: Frame {frame} has invalid values. Cleaning...")
        amplitude = np.nan_to_num(amplitude)
    
    im = axes[i].imshow(amplitude, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    axes[i].set_title(f't = {frame_times[frame]:.2f} s')
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    add_panel_label(axes[i], chr(ord('A') + i))

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Amplitude (a.u.)')

plt.tight_layout(rect=[0, 0, 0.9, 1.0])
plt.subplots_adjust(top=0.8, wspace=0.3)
fig.suptitle('Wave Propagation Timeline', fontsize=12, y=0.95)

# Save in PDF and TIFF formats
plt.savefig('validation_results/Fig3_propagation_timeline_fixed.pdf', dpi=600, bbox_inches='tight')
plt.savefig('validation_results/Fig3_propagation_timeline_fixed.tiff', dpi=600, bbox_inches='tight')
plt.close(fig)

# =============================================
# DIAGNOSTIC VISUALIZATIONS
# =============================================
print("\nCreating detailed diagnostic visualizations...")
rows = 5
cols = 10
fig, axes = plt.subplots(rows, cols, figsize=(15, 8))

all_data = np.abs(np.array(vft_snaps)).flatten()
vmin = 0
vmax = np.percentile(all_data, 95)

frame = 0
for r in range(rows):
    for c in range(cols):
        if frame < snap_frames:
            ax = axes[r, c]
            amplitude = np.abs(vft_snaps[frame])
            ax.imshow(amplitude, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
            ax.set_title(f'{frame}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            frame += 1

plt.tight_layout()
# Save in PDF and TIFF formats, consistent DPI
plt.savefig('validation_results/wave_evolution_diagnostic.pdf', dpi=600, bbox_inches='tight')
plt.savefig('validation_results/wave_evolution_diagnostic.tiff', dpi=600, bbox_inches='tight')
plt.close(fig)

print("\nAll figures regenerated with improved visualization!")
print("Check validation_results folder for the new fixed figures.")
