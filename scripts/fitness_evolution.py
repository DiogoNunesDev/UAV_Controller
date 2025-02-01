import matplotlib.pyplot as plt

file_path = '../txt_files/fitness_evolution.txt'

generations = []
fitness_values = []

with open(file_path, 'r') as file:
    for line in file:
        try:
            gen, fitness = line.split(":")
            
            generation = int(gen.strip().replace("Generation", "").strip())
            fitness_value = float(fitness.strip())
            
            generations.append(generation)
            fitness_values.append(fitness_value)
        except ValueError:
            continue

plt.figure(figsize=(10, 6))
plt.plot(generations, fitness_values, marker='o', linestyle='-', color='b', label='Fitness Progress')

plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness', fontsize=14)
plt.title('Fitness Progress Across Generations', fontsize=16)
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')  
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
