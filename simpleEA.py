import numpy as np
import matplotlib.pyplot as plt
from tools import initialize_population, relative_fitness, calculate_fitness, roulette_wheel_selection, perform_crossover_and_mutation



def evolutionary_algorithm_with_graph(n, B, crossover_prob, mutation_prob, problem_type="OneMax", generations=100):
    population = initialize_population(n, B)
    max_fitness_history = []

    for generation in range(generations):
        fitness_scores = calculate_fitness(population, problem_type)
        max_fitness = np.max(fitness_scores)
        max_fitness_history.append(max_fitness)
        
        if (problem_type == "OneMax" and max_fitness == n) or (problem_type == "OneMin" and max_fitness == 0):
            print(f"Maximum fitness reached at generation {generation}")
            break

        relative_fitnesses = relative_fitness(fitness_scores)
        parents_indices = roulette_wheel_selection(relative_fitnesses)
        offspring = perform_crossover_and_mutation(population, parents_indices, crossover_prob, mutation_prob, n)
        population = offspring

    return population, max_fitness_history


n = 100  # Length of the binary string
B = 1000  # Population size
crossover_prob = 0.5  # Crossover probability
mutation_prob = 0.01  # Mutation probability

final_population, fitness_over_generations = evolutionary_algorithm_with_graph(n, B, crossover_prob, mutation_prob, problem_type="OneMin", generations=500)

plt.plot(fitness_over_generations)
plt.xlabel('Generation')
plt.ylabel('Maximum Fitness')
plt.title('Maximum Fitness per Generation')
plt.show()