import numpy as np
import matplotlib.pyplot as plt
from tools import initialize_population, relative_fitness, calculate_fitness, roulette_wheel_selection, perform_crossover_and_mutation



def evolutionary_algorithm_comparison(n, B, crossover_prob, mutation_prob, elitism_count, problem_type="OneMax", generations=100):
    # Initialize populations
    population_no_elitism = initialize_population(n, B)
    population_elitism = np.copy(population_no_elitism)

    # Lists to track max fitness per generation
    max_fitness_no_elitism = []
    max_fitness_elitism = []

    for generation in range(generations):
        # Fitness calculation for both populations
        fitness_no_elitism = calculate_fitness(population_no_elitism, problem_type)
        fitness_elitism = calculate_fitness(population_elitism, problem_type)

        # Record the maximum fitness of this generation
        max_fitness_no_elitism.append(np.max(fitness_no_elitism))
        max_fitness_elitism.append(np.max(fitness_elitism))

        # Elitism: Carry over the top individuals
        if elitism_count > 0:
            elites_indices = np.argsort(fitness_elitism)[-elitism_count:]
            elites = population_elitism[elites_indices]

        # Selection and generating offspring for both populations
        relative_fitness_no_elitism = relative_fitness(fitness_no_elitism)
        parents_indices_no_elitism = roulette_wheel_selection(relative_fitness_no_elitism)
        offspring_no_elitism = perform_crossover_and_mutation(population_no_elitism, parents_indices_no_elitism, crossover_prob, mutation_prob, n)

        relative_fitness_elitism = relative_fitness(fitness_elitism)
        parents_indices_elitism = roulette_wheel_selection(relative_fitness_elitism)
        offspring_elitism = perform_crossover_and_mutation(population_elitism, parents_indices_elitism, crossover_prob, mutation_prob, n)

        # Survival strategy
        population_no_elitism = offspring_no_elitism

        # Elitism: Add elites back to the population
        if elitism_count > 0:
            non_elites_indices = np.argsort(fitness_elitism)[:-elitism_count]
            population_elitism[non_elites_indices] = offspring_elitism[:len(non_elites_indices)]
            population_elitism[elites_indices] = elites
        else:
            population_elitism = offspring_elitism

    return max_fitness_no_elitism, max_fitness_elitism

# Parameters
n = 100  # Length of the binary string
B = 1000  # Population size
crossover_prob = 0.5  # Crossover probability
mutation_prob = 0.01  # Mutation probability
elitism_count = int(0.1*B)  # Number of elites to carry over
generations = 200  # Number of generations

# Run the algorithm comparison
fitness_no_elitism, fitness_elitism = evolutionary_algorithm_comparison(n, B, crossover_prob, mutation_prob, elitism_count, generations=generations)

# Plotting the graphs
plt.figure(figsize=(12, 6))

# Graph without elitism
plt.subplot(1, 2, 1)
plt.plot(fitness_no_elitism, label='Without Elitism')
plt.xlabel('Generation')
plt.ylabel('Maximum Fitness')
plt.title('Maximum Fitness per Generation (Without Elitism)')
plt.legend()

# Graph with elitism
plt.subplot(1, 2, 2)
plt.plot(fitness_elitism, label='With Elitism')
plt.xlabel('Generation')
plt.ylabel('Maximum Fitness')
plt.title('Maximum Fitness per Generation (With Elitism)')
plt.legend()

plt.tight_layout()
plt.show()

