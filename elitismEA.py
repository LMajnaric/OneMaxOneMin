import numpy as np
import matplotlib.pyplot as plt
from tools import initialize_population, relative_fitness, calculate_fitness, roulette_wheel_selection, perform_crossover_and_mutation



def evolutionary_algorithm_with_elitism(n, B, crossover_prob, mutation_prob, elitism_count, problem_type="OneMax", generations=100):
    # Initialize population
    population = initialize_population(n, B)
    max_fitness_per_generation = []

    for generation in range(generations):
        # Fitness calculation
        fitness_scores = calculate_fitness(population, problem_type)
        max_fitness_per_generation.append(np.max(fitness_scores))

        # Check for maximum fitness
        max_fitness = n if problem_type == "OneMax" else 0
        if max_fitness in fitness_scores:
            print(f"Maximum fitness reached at generation {generation}")
            break

        # Elitism: Carry over the top individuals
        elites_indices = np.argsort(fitness_scores)[-elitism_count:]
        elites = population[elites_indices]

        # Selection and generating offspring
        relative_fitnesses = relative_fitness(fitness_scores)
        parents_indices = roulette_wheel_selection(relative_fitnesses)
        offspring = perform_crossover_and_mutation(population, parents_indices, crossover_prob, mutation_prob, n)

        # Replace least fit individuals with offspring
        non_elites_indices = np.argsort(fitness_scores)[:-elitism_count]
        population[non_elites_indices] = offspring[:len(non_elites_indices)]

        # Add elites back to the population
        population[elites_indices] = elites

    return population, max_fitness_per_generation


n = 100  # Length of the binary string
B = 1000  # Population size
crossover_prob = 0.5  # Crossover probability
mutation_prob = 0.01  # Mutation probability
elitism_count = int(0.1*B)

final_population, fitness_over_generations = evolutionary_algorithm_with_elitism(n, B, crossover_prob, mutation_prob, elitism_count, generations=500)

plt.plot(fitness_over_generations)
plt.xlabel('Generation')
plt.ylabel('Maximum Fitness')
plt.title('Maximum Fitness per Generation')
plt.show()