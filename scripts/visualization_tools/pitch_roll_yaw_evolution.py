import matplotlib.pyplot as plt

file_path = '../txt_files/pitch_roll_yaw_evolution.txt'

pitch_values = []
roll_values = []
yaw_values = []

with open(file_path, 'r') as file:
    for line in file:
        try:
            if 'Pitch' in line and 'Roll' in line and 'Yaw' in line:
                parts = line.split(", ")
                pitch = float(parts[0].split(":")[1].strip())
                roll = float(parts[1].split(":")[1].strip())
                yaw = float(parts[2].split(":")[1].strip())

                pitch_values.append(pitch)
                roll_values.append(roll)
                yaw_values.append(yaw)
        except ValueError:
            continue  

fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plot Pitch evolution
axs[0].plot(pitch_values, color='b', label='Pitch (deg)', marker='o')
axs[0].set_title('Pitch Evolution Over Time', fontsize=16)
axs[0].set_xlabel('Data Point', fontsize=14)
axs[0].set_ylabel('Pitch (degrees)', fontsize=14)
axs[0].grid(True)
axs[0].legend()

# Plot Roll evolution
axs[1].plot(roll_values, color='g', label='Roll (deg)', marker='o')
axs[1].set_title('Roll Evolution Over Time', fontsize=16)
axs[1].set_xlabel('Data Point', fontsize=14)
axs[1].set_ylabel('Roll (degrees)', fontsize=14)
axs[1].grid(True)
axs[1].legend()

# Plot Yaw evolution
axs[2].plot(yaw_values, color='r', label='Yaw (deg)', marker='o')
axs[2].set_title('Yaw Evolution Over Time', fontsize=16)
axs[2].set_xlabel('Data Point', fontsize=14)
axs[2].set_ylabel('Yaw (degrees)', fontsize=14)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
