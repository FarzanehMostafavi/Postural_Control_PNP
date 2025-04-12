import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- Load Force Plate Data ---
# Replace 'force_plate_data.csv' with your actual data file
# Expected columns: Time, Fx, Fy, Fz, Mx, My, Mz
data = pd.read_csv('force_plate_data.csv')

# --- Define Constants ---
h = 0.0  # Vertical distance from force plate surface to origin of moment measurements (in meters)

# --- Calculate Center of Pressure (CoP) ---
# Avoid division by zero by replacing zeros in Fz with NaN
data['Fz'].replace(0, np.nan, inplace=True)

# Compute CoP in X and Y directions
data['CoP_X'] = (-data['My'] - data['Fx'] * h) / data['Fz']
data['CoP_Y'] = (data['Mx'] - data['Fy'] * h) / data['Fz']

# --- Apply Low-Pass Filter to CoP Data ---
def low_pass_filter(signal, cutoff=10, fs=1000, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Apply filter to CoP signals
data['CoP_X_filtered'] = low_pass_filter(data['CoP_X'].fillna(method='ffill'))
data['CoP_Y_filtered'] = low_pass_filter(data['CoP_Y'].fillna(method='ffill'))

# --- Calculate CoP Metrics ---
# Compute CoP path length
cop_diff = np.sqrt(np.diff(data['CoP_X_filtered'])**2 + np.diff(data['CoP_Y_filtered'])**2)
cop_path_length = np.sum(cop_diff)

# Compute mean velocity of CoP
duration = data['Time'].iloc[-1] - data['Time'].iloc[0]
mean_velocity = cop_path_length / duration

# Compute sway area using 95% confidence ellipse
from matplotlib.patches import Ellipse

def compute_ellipse_area(x, y):
    cov = np.cov(x, y)
    eigenvalues, _ = np.linalg.eig(cov)
    # 95% confidence ellipse area
    area = np.pi * np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1]) * 2.4477
    return area

sway_area = compute_ellipse_area(data['CoP_X_filtered'], data['CoP_Y_filtered'])

# --- Output Results ---
print(f"CoP Path Length: {cop_path_length:.2f} mm")
print(f"Mean CoP Velocity: {mean_velocity:.2f} mm/s")
print(f"Sway Area (95% Confidence Ellipse): {sway_area:.2f} mm²")

# --- Visualize CoP Trajectory ---
plt.figure(figsize=(8, 6))
plt.plot(data['CoP_X_filtered'], data['CoP_Y_filtered'], label='CoP Trajectory')
plt.xlabel('CoP X (mm)')
plt.ylabel('CoP Y (mm)')
plt.title('Center of Pressure Trajectory')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
