import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from scipy.ndimage import gaussian_filter
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.transforms import ScaledTranslation

def add_panel_label(ax, label, x_offset=0.01, y_offset=0.98):
    """
    Adds a panel label (A, B, C) to a subplot, handling both 2D and 3D axes.
    """
    if isinstance(ax, Axes3D):
        # Use text2D for 3D plots
        ax.text2D(x_offset, y_offset, label, transform=ax.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='left')
    else:
        # Use standard text for 2D plots
        ax.text(x_offset, y_offset, label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

def improved_wave_visualization():
    """Create improved wave propagation visualizations for VFT"""
    # Create output directory
    if not os.path.exists('validation_results'):
        os.makedirs('validation_results')
    
    # === Global Matplotlib style settings for paper submission ===
    # Set font to Arial/Helvetica
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    
    # Set font sizes
    rcParams['axes.labelsize'] = 9  # 8-10pt for labels
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['axes.titlesize'] = 11 # 10-12pt for titles
    rcParams['legend.fontsize'] = 9
    rcParams['font.size'] = 9
    
    # Define figure dimensions in inches
    SINGLE_COLUMN = (3.5, 3.0)
    DOUBLE_COLUMN = (7.2, 5.0)
    
    # Create a simplified frequency field with clear regions
    Nx, Ny = 200, 100
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 0.5, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Create a frequency field with 3 distinct regions
    f_field = np.ones((Ny, Nx)) * 10.0  # Base frequency
    f_field[:, :Nx//3] = 5.0            # Low frequency (left)
    f_field[:, Nx//3:2*Nx//3] = 10.0    # Medium frequency (middle)
    f_field[:, 2*Nx//3:] = 20.0         # High frequency (right)
    f_field = gaussian_filter(f_field, sigma=3)  # Smooth boundaries
    
    # Calculate propagation speed (v = 1/f)
    v_field = 1.0 / f_field
    
    # Simulation parameters
    dx = 1.0 / Nx
    dt = 0.0005
    steps = 300
    damping = 0.0005
    
    # Initialize wave field with a point source at center-left
    wave_field = np.zeros((Ny, Nx), dtype=np.complex128)
    source_x, source_y = Nx//6, Ny//2
    pulse_width = 5
    for i in range(-pulse_width, pulse_width+1):
        for j in range(-pulse_width, pulse_width+1):
            if 0 <= source_y+i < Ny and 0 <= source_x+j < Nx:
                wave_field[source_y+i, source_x+j] = 1.0
    
    prev_field = wave_field.copy()
    snapshots = []
    
    # Run simulation
    print("Running wave propagation simulation...")
    for step in range(steps):
        if step % 10 == 0:
            snapshots.append(np.abs(wave_field.copy()))
            print(f"Step {step}/{steps}")
        
        # Calculate Laplacian
        laplacian = np.zeros_like(wave_field)
        for i in range(1, Ny-1):
            for j in range(1, Nx-1):
                laplacian[i, j] = (wave_field[i+1, j] + wave_field[i-1, j] + 
                                  wave_field[i, j+1] + wave_field[i, j-1] - 
                                  4*wave_field[i, j]) / (dx*dx)
        
        # Update wave field using wave equation
        next_field = (2*wave_field - prev_field + 
                     dt*dt * v_field*v_field * laplacian - 
                     damping * dt * (wave_field - prev_field))
        
        prev_field = wave_field.copy()
        wave_field = next_field.copy()
    
    # Select frames for visualization
    total_frames = len(snapshots)
    frame_indices = [0, total_frames//5, 2*total_frames//5, 3*total_frames//5, 4*total_frames//5]
    selected_frames = [snapshots[i] for i in frame_indices]
    
    # ===============================
    # 1. CONTOUR PLOT VISUALIZATION
    # ===============================
    print("Creating contour plot visualization...")
    # Use double-column figure size
    fig = plt.figure(figsize=DOUBLE_COLUMN)
    
    # Plot frequency field with contours
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    contour = ax1.contourf(X, Y, f_field, 20, cmap='viridis')
    ax1.set_title('Frequency Field $f(\\mathbf{x})$ with Wave Propagation Speed')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    
    # Overlay propagation speed vectors
    fx, fy = np.gradient(f_field)
    quiver_skip = 10  # Skip points for clearer visualization
    ax1.quiver(X[::quiver_skip, ::quiver_skip], 
              Y[::quiver_skip, ::quiver_skip], 
              -fx[::quiver_skip, ::quiver_skip], 
              -fy[::quiver_skip, ::quiver_skip],
              color='white', alpha=0.8, scale=30)
    
    plt.colorbar(contour, ax=ax1, label='Frequency (Hz)')
    add_panel_label(ax1, 'A')
    
    # Plot wave propagation contours at different times
    panel_labels = 'BCDEF'
    for i, (frame, idx) in enumerate(zip(selected_frames, frame_indices)):
        row = 1 + i // 2
        col = i % 2
        
        ax = plt.subplot2grid((4, 2), (row, col))
        
        ax.contourf(X, Y, f_field, 5, cmap='viridis', alpha=0.3)
        levels = np.linspace(0, np.max(frame)*0.8, 15)
        cs = ax.contour(X, Y, frame, levels, cmap='inferno')
        plt.setp(cs, linewidth=0.8)
        
        time = idx * 10 * dt
        ax.set_title(f't = {time:.3f} s')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        add_panel_label(ax, panel_labels[i])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.4)
    plt.suptitle('Wave Propagation in VFT: Contour Visualization', y=0.98)
    plt.savefig('validation_results/vft_wave_contours.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('validation_results/vft_wave_contours.tiff', dpi=600, bbox_inches='tight')
    
    # ===============================
    # 2. 3D SURFACE VISUALIZATION
    # ===============================
    print("Creating 3D surface visualization...")
    # Use double-column figure size
    fig = plt.figure(figsize=DOUBLE_COLUMN)
    
    best_frame = snapshots[total_frames//3]
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, best_frame, cmap='inferno', 
                           linewidth=0, antialiased=True)
    ax1.set_title('Wave Amplitude (3D)')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('y (mm)')
    ax1.set_zlabel('Amplitude (a.u.)')
    add_panel_label(ax1, 'A')
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, f_field, cmap='viridis',
                           linewidth=0, antialiased=True)
    ax2.set_title('Frequency Field $f(\\mathbf{x})$ (3D)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('y (mm)')
    ax2.set_zlabel('Frequency (Hz)')
    add_panel_label(ax2, 'B')
    
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(best_frame, origin='lower', cmap='inferno', aspect='auto',
                     extent=[x.min(), x.max(), y.min(), y.max()])
    ax3.set_title('Wave Amplitude (2D)')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('y (mm)')
    plt.colorbar(im3, ax=ax3, label='Amplitude (a.u.)')
    add_panel_label(ax3, 'C')
    
    ax4 = fig.add_subplot(2, 2, 4)
    unique_freqs = np.linspace(np.min(f_field), np.max(f_field), 100)
    speeds = 1.0 / unique_freqs
    ax4.plot(unique_freqs, speeds, 'r-', linewidth=2)
    ax4.set_title('Propagation Speed vs. Frequency')
    ax4.set_xlabel('Frequency $f$ (Hz)')
    ax4.set_ylabel('Propagation Speed $v = 1/f$ (m/s)')
    ax4.grid(True, alpha=0.3)
    
    ax4.text(0.5, 0.9, '$v(\\mathbf{x}) = \\frac{1}{f(\\mathbf{x})}$', 
            transform=ax4.transAxes, ha='center', va='top', fontsize=10)
    add_panel_label(ax4, 'D')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.4)
    plt.suptitle('Vibrational Field Theory: 3D Visualization', y=0.98)
    plt.savefig('validation_results/vft_wave_3d.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('validation_results/vft_wave_3d.tiff', dpi=600, bbox_inches='tight')
    
    # ===============================
    # 3. SPACE-TIME VISUALIZATION
    # ===============================
    print("Creating space-time visualization...")
    # Use double-column figure size
    fig = plt.figure(figsize=DOUBLE_COLUMN)
    
    middle_row = Ny // 2
    spacetime = np.array([snapshot[middle_row, :] for snapshot in snapshots])
    
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.plot(x, f_field[middle_row, :], 'b-', linewidth=2)
    ax1.set_title('Frequency Profile Along Middle Row')
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.grid(True, alpha=0.3)
    add_panel_label(ax1, 'A')
    
    ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    times = np.arange(len(snapshots)) * 10 * dt
    im = ax2.imshow(spacetime, cmap='inferno', 
                    extent=[x.min(), x.max(), times[-1], 0],
                    aspect='auto', origin='upper')
    ax2.set_title('Space-Time Diagram')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Time (s)')
    plt.colorbar(im, ax=ax2, label='Amplitude (a.u.)')
    add_panel_label(ax2, 'B')
    
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax3.plot(x, v_field[middle_row, :], 'r-', linewidth=2)
    ax3.set_title('Propagation Speed Profile Along Middle Row')
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.grid(True, alpha=0.3)
    add_panel_label(ax3, 'C')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.4)
    plt.suptitle('VFT Wave Propagation: Space-Time Analysis', y=0.98)
    plt.savefig('validation_results/vft_wave_spacetime.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('validation_results/vft_wave_spacetime.tiff', dpi=600, bbox_inches='tight')
    
    plt.show()

# Run the improved visualization
if __name__ == '__main__':
    improved_wave_visualization()
