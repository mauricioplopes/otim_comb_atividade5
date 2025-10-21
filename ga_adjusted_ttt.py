#!/usr/bin/env python3
"""
Genetic Algorithm for MAX-SC-QBF Problem - Ajustado para TTT-Plots
Maximização de Função Binária Quadrática com Restrições de Cobertura de Conjuntos

Uso:
    python ga_adjusted.py <instancia> <tempo_limite> [opcoes]

Opções:
    --generations N          Número de gerações (padrão: 1000)
    --popsize N             Tamanho da população (padrão: 100)
    --mutation RATE         Taxa de mutação (padrão: 0.01)
    --seed N                Seed aleatória (padrão: 0)
    --strategy STRATEGY     Estratégia: random, lhc, greedy (padrão: random)
    --adaptive-mutation     Ativa mutação adaptativa
    --target VALOR          Valor alvo para TTT-plot (opcional)
    --verbose               Modo verboso
    --quiet                 Modo silencioso
    --csv ARQUIVO           Arquivo CSV de saída (formato TTT-plot)

Exemplo:
    python ga_adjusted.py instances/qbf_sc/instance-01.txt 600 --seed 0 --target 354 --csv results.csv
"""

import sys
import os
import random
import time
import argparse
from typing import List, Set


# =============================================================================
# SOLUTION CLASS
# =============================================================================

class Solution(list):
    """Solution class that extends list to store solution elements."""
    
    def __init__(self, elements=None):
        super().__init__()
        self.cost = float('inf')
        if elements is not None:
            self.extend(elements)
    
    def copy(self):
        """Create a deep copy of the solution."""
        new_sol = Solution(list(self))
        new_sol.cost = self.cost
        return new_sol
    
    def __str__(self):
        return f"Solution: cost=[{self.cost:.2f}], size=[{len(self)}], elements={list(self)[:10]}{'...' if len(self) > 10 else ''}"


# =============================================================================
# QBF-SC EVALUATOR
# =============================================================================

class QBF_SC:
    """
    QBF with Set Cover constraints for MAXIMIZATION.
    Returns POSITIVE values as this is a MAXIMIZATION problem.
    """
    
    def __init__(self, filename):
        self.sets = []
        self.universe = set()
        self.size = None
        self.A = None
        self.variables = None
        self.size = self._read_input(filename)
        self.variables = [0.0] * self.size
    
    def _read_input(self, filename):
        """Read QBF-SC instance from file."""
        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                raise ValueError("Empty file")
            
            n = int(lines[0])
            self.A = [[0.0 for _ in range(n)] for _ in range(n)]
            
            line_idx = 2
            self.sets = []
            self.universe = set(range(1, n + 1))
            
            for i in range(n):
                if line_idx >= len(lines):
                    raise ValueError(f"Missing set definition for variable {i}")
                
                elements_str = lines[line_idx].split()
                if elements_str:
                    elements = set(int(elem) for elem in elements_str)
                    if not elements.issubset(self.universe):
                        invalid = elements - self.universe
                        raise ValueError(f"Set {i} contains invalid elements: {invalid}")
                    self.sets.append(elements)
                else:
                    self.sets.append(set())
                
                line_idx += 1
            
            for i in range(n):
                if line_idx >= len(lines):
                    raise ValueError(f"Missing matrix row {i}")
                
                values = list(map(float, lines[line_idx].split()))
                expected_elements = n - i
                
                if len(values) != expected_elements:
                    raise ValueError(f"Row {i}: expected {expected_elements} elements, got {len(values)}")
                
                for j, val in enumerate(values):
                    col_idx = i + j
                    self.A[i][col_idx] = val
                    if col_idx != i:
                        self.A[col_idx][i] = 0.0
                
                line_idx += 1
            
            return n
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Instance file '{filename}' not found")
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {e}")
    
    def get_domain_size(self):
        return self.size
    
    def reset_variables(self):
        self.variables = [0.0] * self.size
    
    def set_variables(self, sol):
        self.reset_variables()
        if sol:
            for elem in sol:
                self.variables[elem] = 1.0
    
    def evaluate(self, sol):
        self.set_variables(sol)
        sol.cost = self.evaluate_qbf()
        return sol.cost
    
    def evaluate_qbf(self):
        """Compute QBF value for MAXIMIZATION (positive values)."""
        total = 0.0
        for i in range(self.size):
            aux = 0.0
            for j in range(self.size):
                aux += self.variables[j] * self.A[i][j]
            total += aux * self.variables[i]
        return total  # Return POSITIVE value for maximization
    
    def is_feasible(self, sol) -> bool:
        """Check if solution satisfies set cover constraints."""
        covered = set()
        for var_idx in sol:
            if 0 <= var_idx < len(self.sets):
                covered.update(self.sets[var_idx])
        return self.universe.issubset(covered)
    
    def get_uncovered_elements(self, sol) -> Set[int]:
        covered = set()
        for var_idx in sol:
            if 0 <= var_idx < len(self.sets):
                covered.update(self.sets[var_idx])
        return self.universe - covered
    
    def get_coverage_count(self, sol) -> dict:
        coverage = {elem: 0 for elem in self.universe}
        for var_idx in sol:
            if 0 <= var_idx < len(self.sets):
                for elem in self.sets[var_idx]:
                    if elem in coverage:
                        coverage[elem] += 1
        return coverage
    
    def get_removable_variables(self, sol) -> List[int]:
        if not self.is_feasible(sol):
            return []
        
        coverage = self.get_coverage_count(sol)
        removable = []
        
        for var_idx in sol:
            if var_idx >= len(self.sets):
                continue
            
            can_remove = True
            for elem in self.sets[var_idx]:
                if coverage.get(elem, 0) <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                removable.append(var_idx)
        
        return removable


# =============================================================================
# GENETIC ALGORITHM FOR QBF-SC
# =============================================================================

class GA_QBF_SC:
    """Genetic Algorithm for QBF with Set Cover constraints."""
    
    def __init__(self, qbf_sc, generations, pop_size, mutation_rate, time_limit,
                 seed=0, population_strategy='random', adaptive_mutation=False, 
                 target_value=None, verbose=True):
        self.obj_function = qbf_sc
        self.qbf_sc = qbf_sc
        self.generations = generations
        self.pop_size = pop_size
        self.chromosome_size = qbf_sc.get_domain_size()
        self.mutation_rate = mutation_rate
        self.time_limit = time_limit
        self.population_strategy = population_strategy.lower()
        self.adaptive_mutation = adaptive_mutation
        self.target_value = target_value
        self.verbose = verbose
        
        self.best_cost = None
        self.best_sol = None
        self.best_chromosome = None
        
        # Estatísticas
        self.start_time = None
        self.time_to_target = None
        self.generations_completed = 0
        
        random.seed(seed)
        
        valid_strategies = ['random', 'lhc', 'greedy']
        if self.population_strategy not in valid_strategies:
            raise ValueError(f"Invalid population_strategy: {self.population_strategy}. "
                           f"Must be one of: {valid_strategies}")
    
    def create_empty_sol(self):
        sol = Solution()
        sol.cost = 0.0
        return sol
    
    def decode(self, chromosome):
        solution = self.create_empty_sol()
        for locus in range(len(chromosome)):
            if chromosome[locus] == 1:
                solution.append(locus)
        self.obj_function.evaluate(solution)
        return solution
    
    def generate_random_chromosome(self):
        chromosome = [random.randint(0, 1) for _ in range(self.chromosome_size)]
        self.repair_chromosome(chromosome)
        return chromosome
    
    def fitness(self, chromosome):
        sol = self.decode(chromosome)
        if not self.qbf_sc.is_feasible(sol):
            uncovered = self.qbf_sc.get_uncovered_elements(sol)
            penalty = len(uncovered) * 1000.0
            return sol.cost - penalty
        return sol.cost
    
    def mutate_gene(self, chromosome, locus):
        chromosome[locus] = 1 - chromosome[locus]
        if chromosome[locus] == 0:
            sol = self.decode(chromosome)
            if not self.qbf_sc.is_feasible(sol):
                self.repair_chromosome(chromosome)
    
    def repair_chromosome(self, chromosome):
        covered = set()
        for var_idx in range(self.chromosome_size):
            if chromosome[var_idx] == 1:
                covered.update(self.qbf_sc.sets[var_idx])
        
        uncovered = self.qbf_sc.universe - covered
        
        while uncovered:
            best_var = None
            best_count = 0
            
            for var_idx in range(self.chromosome_size):
                if chromosome[var_idx] == 1:
                    continue
                
                count = len(self.qbf_sc.sets[var_idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_var = var_idx
            
            if best_var is None:
                break
            
            chromosome[best_var] = 1
            uncovered -= self.qbf_sc.sets[best_var]
    
    def initialize_population(self):
        if self.population_strategy == 'lhc':
            return self.initialize_population_lhc()
        elif self.population_strategy == 'greedy':
            return self.initialize_population_greedy()
        else:
            return self.initialize_population_random()
    
    def initialize_population_random(self):
        population = []
        for _ in range(self.pop_size):
            population.append(self.generate_random_chromosome())
        return population
    
    def initialize_population_lhc(self):
        population = []
        for i in range(self.pop_size):
            chromosome = [0] * self.chromosome_size
            for j in range(self.chromosome_size):
                stratum_index = (i + j * self.pop_size) % self.pop_size
                threshold = (stratum_index + random.random()) / self.pop_size
                chromosome[j] = 1 if threshold > 0.5 else 0
            self.repair_chromosome(chromosome)
            population.append(chromosome)
        return population
    
    def initialize_population_greedy(self):
        population = []
        for _ in range(self.pop_size):
            population.append(self.generate_greedy_random_chromosome())
        return population
    
    def generate_greedy_random_chromosome(self):
        chromosome = [0] * self.chromosome_size
        uncovered = self.qbf_sc.universe.copy()
        
        while uncovered:
            best_var = None
            best_count = 0
            
            for var_idx in range(self.chromosome_size):
                if chromosome[var_idx] == 1:
                    continue
                count = len(self.qbf_sc.sets[var_idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_var = var_idx
            
            if best_var is None:
                break
            
            chromosome[best_var] = 1
            uncovered -= self.qbf_sc.sets[best_var]
        
        for var_idx in range(self.chromosome_size):
            if chromosome[var_idx] == 0 and random.random() < 0.3:
                chromosome[var_idx] = 1
        
        return chromosome
    
    def get_best_chromosome(self, population):
        best_fitness = float('-inf')
        best_chromosome = None
        for chromosome in population:
            fit = self.fitness(chromosome)
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = chromosome
        return best_chromosome
    
    def get_worse_chromosome(self, population):
        worse_fitness = float('inf')
        worse_chromosome = None
        for chromosome in population:
            fit = self.fitness(chromosome)
            if fit < worse_fitness:
                worse_fitness = fit
                worse_chromosome = chromosome
        return worse_chromosome
    
    def select_parents(self, population):
        parents = []
        for _ in range(self.pop_size):
            idx1 = random.randint(0, self.pop_size - 1)
            parent1 = population[idx1]
            idx2 = random.randint(0, self.pop_size - 1)
            parent2 = population[idx2]
            
            if self.fitness(parent1) > self.fitness(parent2):
                parents.append(parent1[:])
            else:
                parents.append(parent2[:])
        return parents
    
    def crossover(self, parents):
        offsprings = []
        for i in range(0, self.pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            crosspoint1 = random.randint(0, self.chromosome_size)
            crosspoint2 = crosspoint1 + random.randint(0, self.chromosome_size - crosspoint1)
            
            offspring1 = []
            offspring2 = []
            
            for j in range(self.chromosome_size):
                if j >= crosspoint1 and j < crosspoint2:
                    offspring1.append(parent2[j])
                    offspring2.append(parent1[j])
                else:
                    offspring1.append(parent1[j])
                    offspring2.append(parent2[j])
            
            self.repair_chromosome(offspring1)
            self.repair_chromosome(offspring2)
            
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        
        return offsprings
    
    def get_mutation_rate(self, generation):
        if not self.adaptive_mutation:
            return self.mutation_rate
        
        initial_rate = 0.1
        final_rate = 0.001
        progress = generation / self.generations if self.generations > 0 else 0
        current_rate = initial_rate * (1 - progress) + final_rate * progress
        return current_rate
    
    def mutate(self, offsprings, generation=None):
        if generation is not None and self.adaptive_mutation:
            mutation_rate = self.get_mutation_rate(generation)
        else:
            mutation_rate = self.mutation_rate
        
        for chromosome in offsprings:
            for locus in range(self.chromosome_size):
                if random.random() < mutation_rate:
                    self.mutate_gene(chromosome, locus)
        
        return offsprings
    
    def select_population(self, offsprings):
        worse = self.get_worse_chromosome(offsprings)
        if self.fitness(worse) < self.fitness(self.best_chromosome):
            offsprings.remove(worse)
            offsprings.append(self.best_chromosome[:])
        return offsprings
    
    def solve(self):
        self.start_time = time.time()
        
        population = self.initialize_population()
        
        self.best_chromosome = self.get_best_chromosome(population)
        self.best_sol = self.decode(self.best_chromosome)
        
        if self.verbose:
            print(f"(Gen. 0) BestSol = {self.best_sol}")
            print(f"         Feasible: {self.qbf_sc.is_feasible(self.best_sol)}")
            print(f"         Strategy: {self.population_strategy.upper()}")
            if self.adaptive_mutation:
                print(f"         Adaptive Mutation: ENABLED (rate: {self.get_mutation_rate(0):.4f})")
            else:
                print(f"         Mutation Rate: {self.mutation_rate:.4f} (fixed)")
        
        for g in range(1, self.generations + 1):
            self.generations_completed = g
            
            # Verifica tempo limite
            elapsed = time.time() - self.start_time
            if elapsed >= self.time_limit:
                if self.verbose:
                    print(f"\nTempo limite atingido na geração {g}")
                break
            
            parents = self.select_parents(population)
            offsprings = self.crossover(parents)
            mutants = self.mutate(offsprings, generation=g)
            population = self.select_population(mutants)
            
            best_chromosome = self.get_best_chromosome(population)
            
            if self.fitness(best_chromosome) > self.fitness(self.best_chromosome):
                self.best_chromosome = best_chromosome
                self.best_sol = self.decode(best_chromosome)
                
                # Verifica se atingiu o alvo
                if self.target_value is not None and self.time_to_target is None:
                    if self.best_sol.cost >= self.target_value:
                        self.time_to_target = time.time() - self.start_time
                        if self.verbose:
                            print(f"\n✓ Alvo {self.target_value} atingido na geração {g}")
                            print(f"  Tempo até alvo: {self.time_to_target:.2f}s")
                
                if self.verbose:
                    is_feasible = self.qbf_sc.is_feasible(self.best_sol)
                    print(f"(Gen. {g}) BestSol = {self.best_sol}")
                    print(f"         Feasible: {is_feasible}")
                    if self.adaptive_mutation:
                        print(f"         Mutation Rate: {self.get_mutation_rate(g):.4f}")
        
        return self.best_sol


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Genetic Algorithm for MAX-SC-QBF Problem',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
  python ga_adjusted.py instances/qbf_sc/instance-01.txt 600 --seed 0 --target 354 --csv results.csv
        """
    )
    
    parser.add_argument('instance', help='Arquivo da instância')
    parser.add_argument('time_limit', type=float, help='Tempo limite em segundos')
    
    parser.add_argument('--generations', type=int, default=1000,
                       help='Número de gerações (padrão: 1000)')
    parser.add_argument('--popsize', type=int, default=100,
                       help='Tamanho da população (padrão: 100)')
    parser.add_argument('--mutation', type=float, default=0.01,
                       help='Taxa de mutação (padrão: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed aleatória (padrão: 0)')
    parser.add_argument('--strategy', choices=['random', 'lhc', 'greedy'],
                       default='random',
                       help='Estratégia de inicialização (padrão: random)')
    parser.add_argument('--adaptive-mutation', action='store_true',
                       help='Ativa mutação adaptativa')
    parser.add_argument('--target', type=float, default=None,
                       help='Valor alvo para TTT-plot (opcional)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Modo verboso')
    parser.add_argument('--quiet', action='store_true',
                       help='Modo silencioso')
    parser.add_argument('--csv', type=str, default=None,
                       help='Arquivo CSV de saída (formato TTT-plot)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.instance):
        print(f"Error: Instance file '{args.instance}' not found.")
        sys.exit(1)
    
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("=" * 70)
        print("Genetic Algorithm for MAX-SC-QBF Problem")
        print("=" * 70)
        print(f"Instance:      {args.instance}")
        print(f"Time Limit:    {args.time_limit}s")
        print(f"Generations:   {args.generations}")
        print(f"Pop Size:      {args.popsize}")
        print(f"Mutation Rate: {args.mutation}")
        print(f"Seed:          {args.seed}")
        print(f"Strategy:      {args.strategy}")
        print(f"Adaptive Mut:  {'Yes' if args.adaptive_mutation else 'No'}")
        if args.target:
            print(f"Target Value:  {args.target}")
        print("=" * 70)
        print()
    
    try:
        start_time = time.time()
        
        qbf_sc = QBF_SC(args.instance)
        ga = GA_QBF_SC(
            qbf_sc,
            args.generations,
            args.popsize,
            args.mutation,
            args.time_limit,
            args.seed,
            args.strategy,
            args.adaptive_mutation,
            args.target,
            verbose
        )
        
        best_sol = ga.solve()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Extract instance name
        instance_name = os.path.basename(args.instance)
        
        # Build algorithm name with parameters
        algorithm_name = f"GA_pop{args.popsize}_mut{args.mutation}_gen{args.generations}"
        if args.strategy != 'random':
            algorithm_name += f"_{args.strategy}"
        if args.adaptive_mutation:
            algorithm_name += "_adaptive"
        
        # Determine hit
        hit = 1 if (args.target is not None and ga.time_to_target is not None) else 0
        
        # Time to target
        time_sec = ga.time_to_target if ga.time_to_target is not None else total_time
        
        # Output
        if not verbose:
            print(f"Custo: {best_sol.cost:.2f}")
            print(f"Tempo: {total_time:.2f}")
            if ga.time_to_target is not None:
                print(f"TempoAlvo: {ga.time_to_target:.2f}")
            print(f"Geracoes: {ga.generations_completed}")
            print(f"Tamanho: {len(best_sol)}")
            print(f"Factivel: {qbf_sc.is_feasible(best_sol)}")
        else:
            print()
            print("=" * 70)
            print("Final Results")
            print("=" * 70)
            print(f"Best Solution: {best_sol}")
            print(f"Cost:          {best_sol.cost:.2f}")
            print(f"Size:          {len(best_sol)} variables selected")
            print(f"Feasible:      {qbf_sc.is_feasible(best_sol)}")
            print(f"Time:          {total_time:.2f} seconds")
            print(f"Generations:   {ga.generations_completed}")
            if ga.time_to_target is not None:
                print(f"Time to target: {ga.time_to_target:.2f}s")
            print("=" * 70)
        
        # CSV output for TTT-plot
        if args.csv:
            import csv
            file_exists = os.path.isfile(args.csv)
            
            with open(args.csv, 'a', newline='') as csvfile:
                fieldnames = ['instance', 'target', 'algorithm', 'seed', 'time_sec', 'hit', 'best_cost']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'instance': instance_name,
                    'target': args.target if args.target is not None else '',
                    'algorithm': algorithm_name,
                    'seed': args.seed,
                    'time_sec': f'{time_sec:.4f}',
                    'hit': hit,
                    'best_cost': f'{best_sol.cost:.2f}'
                })
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
