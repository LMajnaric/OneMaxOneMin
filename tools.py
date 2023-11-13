import numpy as np



def initialize_population(n, B):
    return np.random.randint(2, size=(B, n))

def calculate_fitness(population, problem_type="OneMax"):
    n = population.shape[1]
    if problem_type == "OneMax":
        return np.sum(population, axis=1)
    else:  # OneMin
        return n - np.sum(population, axis=1)

def relative_fitness(fitness_scores):
    total_fitness = np.sum(fitness_scores)
    return fitness_scores / total_fitness

def roulette_wheel_selection(relative_fitnesses):
    cumulative_sum = np.cumsum(relative_fitnesses)
    random_values = np.random.rand(len(relative_fitnesses))
    parents_indices = np.searchsorted(cumulative_sum, random_values)
    return parents_indices

def perform_crossover_and_mutation(population, parents_indices, crossover_prob, mutation_prob, n):
    offspring = np.empty_like(population)
    for i in range(0, len(parents_indices), 2):
        parent1, parent2 = population[parents_indices[i]], population[parents_indices[i+1]]

        # Crossover
        if np.random.rand() < crossover_prob:
            crossover_point = np.random.randint(1, n)
            offspring[i] = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring[i+1] = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        else:
            offspring[i], offspring[i+1] = parent1, parent2

    # Mutation
    for i in range(len(offspring)):
        for j in range(n):
            if np.random.rand() < mutation_prob:
                offspring[i][j] = 1 - offspring[i][j]
    
    return offspring
