#!/usr/bin/env python3
"""
GRASP for MAX-SC-QBF Problem - Ajustado para TTT-Plots
Maximização de Função Binária Quadrática com Restrições de Cobertura de Conjuntos

Uso:
    python grasp_adjusted_ttt.py <instancia> <tempo_limite> [opcoes]

Opções:
    --alpha VALOR               Parâmetro alpha para RCL (padrão: 0.1)
    --construction METHOD       Método de construção: standard, random_plus_greedy, sampled_greedy (padrão: standard)
    --local-search METHOD       Método de busca local: first_improving, best_improving (padrão: first_improving)
    --max-iter-no-improvement N Máximo de iterações sem melhoria (padrão: 1000)
    --seed N                    Seed aleatória (padrão: 0)
    --target VALOR              Valor alvo para TTT-plot (opcional)
    --verbose                   Modo verboso
    --quiet                     Modo silencioso
    --csv ARQUIVO               Arquivo CSV de saída (formato TTT-plot)

Exemplo:
    python grasp_adjusted_ttt.py instances/qbf_sc/instance-01.txt 600 --seed 0 --target 2267 --csv results.csv
"""

import sys
import time
import random
import copy
import argparse
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Generic, TypeVar, Optional


# ============================================================================
# SOLUTION CLASS
# ============================================================================

E = TypeVar('E')

class Solution(Generic[E]):
    """Solution class that stores elements and cost"""
    
    def __init__(self, elements_or_solution=None, cost: float = None):
        if isinstance(elements_or_solution, Solution):
            self.elements = copy.deepcopy(elements_or_solution.elements)
            self.cost = elements_or_solution.cost
        elif elements_or_solution is not None:
            self.elements = elements_or_solution if elements_or_solution is not None else []
            self.cost = cost if cost is not None else float('inf')
        else:
            self.elements = []
            self.cost = float('inf')
    
    def add(self, element: E):
        self.elements.append(element)
    
    def remove(self, element: E):
        if element in self.elements:
            self.elements.remove(element)
    
    def __contains__(self, element: E):
        return element in self.elements
    
    def __iter__(self):
        return iter(self.elements)
    
    def __len__(self):
        return len(self.elements)
    
    def is_empty(self):
        return len(self.elements) == 0
    
    def size(self):
        return len(self.elements)
    
    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self.elements)}], elements={self.elements}"


# ============================================================================
# EVALUATOR INTERFACE
# ============================================================================

class Evaluator(ABC, Generic[E]):
    """Abstract evaluator interface for optimization problems"""
    
    @abstractmethod
    def get_domain_size(self) -> int:
        pass
    
    @abstractmethod
    def evaluate(self, solution: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_insertion_cost(self, element: E, solution: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_removal_cost(self, element: E, solution: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def evaluate_exchange_cost(self, elem_in: E, elem_out: E, solution: Solution[E]) -> float:
        pass
    
    @abstractmethod
    def is_feasible(self, solution: Solution[E]) -> bool:
        pass


# ============================================================================
# QBF_SC EVALUATOR
# ============================================================================

class QBF_SC(Evaluator[int]):
    """Quadratic Binary Function with Set Cover constraints"""
    
    def __init__(self, filename: str):
        self.size, self.A, self.subsets = self.read_input(filename)
        self.variables = [0.0] * self.size
    
    def read_input(self, filename: str):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        line_idx = 0
        n = int(lines[line_idx])
        line_idx += 1
        
        subset_sizes = list(map(int, lines[line_idx].split()))
        line_idx += 1
        
        if len(subset_sizes) != n:
            raise ValueError(f"Expected {n} subset sizes, got {len(subset_sizes)}")
        
        subsets = {}
        for i in range(n):
            if subset_sizes[i] > 0:
                if line_idx >= len(lines):
                    raise ValueError(f"Missing elements for subset {i}")
                subset_elements = list(map(int, lines[line_idx].split()))
                if len(subset_elements) != subset_sizes[i]:
                    raise ValueError(f"Subset {i} should have {subset_sizes[i]} elements")
                subsets[i] = set(elem - 1 for elem in subset_elements if elem > 0)
                line_idx += 1
            else:
                subsets[i] = set()
        
        A = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            if line_idx >= len(lines):
                raise ValueError(f"Missing matrix row {i}")
            
            row_elements = list(map(float, lines[line_idx].split()))
            expected_elements = n - i
            
            if len(row_elements) != expected_elements:
                raise ValueError(f"Row {i} should have {expected_elements} elements")
            
            for j, val in enumerate(row_elements):
                col_idx = i + j
                if col_idx < n:
                    A[i][col_idx] = val
            
            line_idx += 1
        
        return n, A, subsets
    
    def get_domain_size(self) -> int:
        return self.size
    
    def set_variables(self, solution: Solution[int]):
        self.reset_variables()
        if not solution.is_empty():
            for elem in solution:
                if 0 <= elem < self.size:
                    self.variables[elem] = 1.0
    
    def reset_variables(self):
        self.variables = [0.0] * self.size
    
    def evaluate(self, solution: Solution[int]) -> float:
        self.set_variables(solution)
        
        if not self.is_feasible(solution):
            solution.cost = float('-inf')
            return solution.cost
        
        solution.cost = self.evaluate_QBF()
        return solution.cost
    
    def evaluate_QBF(self) -> float:
        total = 0.0
        for i in range(self.size):
            for j in range(self.size):
                total += self.variables[i] * self.variables[j] * self.A[i][j]
        return total
    
    def is_feasible(self, solution: Solution[int]) -> bool:
        covered_variables = set()
        for subset_idx in solution:
            if subset_idx in self.subsets:
                covered_variables.update(self.subsets[subset_idx])
        required_coverage = set(range(self.size))
        return required_coverage.issubset(covered_variables)
    
    def get_uncovered_variables(self, solution: Solution[int]) -> Set[int]:
        covered_variables = set()
        for subset_idx in solution:
            if subset_idx in self.subsets:
                covered_variables.update(self.subsets[subset_idx])
        all_variables = set(range(self.size))
        return all_variables - covered_variables
    
    def evaluate_insertion_cost(self, elem: int, solution: Solution[int]) -> float:
        if elem in solution:
            return 0.0
        self.set_variables(solution)
        return self.evaluate_insertion_QBF(elem)
    
    def evaluate_insertion_QBF(self, i: int) -> float:
        if self.variables[i] == 1:
            return 0.0
        return self.evaluate_contribution_QBF(i)
    
    def evaluate_removal_cost(self, elem: int, solution: Solution[int]) -> float:
        if elem not in solution:
            return 0.0
        
        temp_sol = Solution(list(solution.elements))
        temp_sol.remove(elem)
        if not self.is_feasible(temp_sol):
            return float('-inf')
        
        self.set_variables(solution)
        return self.evaluate_removal_QBF(elem)
    
    def evaluate_removal_QBF(self, i: int) -> float:
        if self.variables[i] == 0:
            return 0.0
        return -self.evaluate_contribution_QBF(i)
    
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, solution: Solution[int]) -> float:
        temp_sol = Solution(list(solution.elements))
        temp_sol.remove(elem_out)
        temp_sol.add(elem_in)
        if not self.is_feasible(temp_sol):
            return float('-inf')
        
        self.set_variables(solution)
        return self.evaluate_exchange_QBF(elem_in, elem_out)
    
    def evaluate_exchange_QBF(self, elem_in: int, elem_out: int) -> float:
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1:
            return self.evaluate_removal_QBF(elem_out)
        if self.variables[elem_out] == 0:
            return self.evaluate_insertion_QBF(elem_in)
        
        cost = 0.0
        cost += self.evaluate_contribution_QBF(elem_in)
        cost -= self.evaluate_contribution_QBF(elem_out)
        
        # Subtract interaction between elem_in and elem_out
        if elem_in < elem_out:
            cost -= self.A[elem_in][elem_out]
        else:
            cost -= self.A[elem_out][elem_in]
        
        return cost
    
    def evaluate_contribution_QBF(self, i: int) -> float:
        total = 0.0
        for j in range(self.size):
            if i != j:
                if i < j:
                    total += self.variables[j] * self.A[i][j]
                else:
                    total += self.variables[j] * self.A[j][i]
        total += self.A[i][i]
        return total


# ============================================================================
# ABSTRACT GRASP
# ============================================================================

class AbstractGRASP(ABC, Generic[E]):
    """Abstract GRASP metaheuristic class"""
    
    def __init__(self, obj_function: Evaluator[E], alpha: float, iterations: int, 
                 construction_method: str = "standard", local_search_method: str = "first_improving",
                 max_iter_no_improvement: int = 1000, verbose: bool = True):
        self.obj_function = obj_function
        self.alpha = alpha
        self.iterations = iterations
        self.construction_method = construction_method
        self.local_search_method = local_search_method
        self.max_iter_no_improvement = max_iter_no_improvement
        self.verbose = verbose
        self.best_cost = float('-inf')
        self.cost = float('-inf')
        self.best_sol: Optional[Solution[E]] = None
        self.sol: Optional[Solution[E]] = None
        self.CL: List[E] = []
        self.RCL: List[E] = []
        self.start_time = None
        self.time_limit = None
    
    @abstractmethod
    def make_CL(self) -> List[E]:
        pass
    
    @abstractmethod
    def make_RCL(self) -> List[E]:
        pass
    
    @abstractmethod
    def update_CL(self):
        pass
    
    @abstractmethod
    def create_empty_sol(self) -> Solution[E]:
        pass
    
    @abstractmethod
    def local_search(self) -> Solution[E]:
        pass
    
    @abstractmethod
    def make_feasible(self):
        pass
    
    def time_remaining(self) -> float:
        if self.time_limit is None:
            return float('inf')
        return self.time_limit - (time.time() - self.start_time)
    
    def constructive_heuristic(self) -> Solution[E]:
        if self.construction_method == "random_plus_greedy":
            return self.random_plus_greedy_construction()
        elif self.construction_method == "sampled_greedy":
            return self.sampled_greedy_construction()
        else:
            return self.standard_constructive_heuristic()
    
    def standard_constructive_heuristic(self) -> Solution[E]:
        self.CL = self.make_CL()
        self.RCL = self.make_RCL()
        self.sol = self.create_empty_sol()
        self.cost = float('-inf')
        
        while not self.constructive_stop_criteria():
            if self.time_remaining() <= 0:
                break
            
            max_cost = float('-inf')
            min_cost = float('inf')
            self.cost = self.obj_function.evaluate(self.sol)
            self.update_CL()
            
            if not self.CL:
                break
            
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost < min_cost:
                    min_cost = delta_cost
                if delta_cost > max_cost:
                    max_cost = delta_cost
            
            self.RCL.clear()
            threshold = max_cost - self.alpha * (max_cost - min_cost)
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost >= threshold:
                    self.RCL.append(c)
            
            if self.RCL:
                rnd_index = random.randint(0, len(self.RCL) - 1)
                in_cand = self.RCL[rnd_index]
                self.CL.remove(in_cand)
                self.sol.add(in_cand)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def random_plus_greedy_construction(self) -> Solution[E]:
        self.CL = self.make_CL()
        self.sol = self.create_empty_sol()
        
        num_random = max(1, int(0.3 * len(self.CL)))
        for _ in range(min(num_random, len(self.CL))):
            if not self.CL or self.time_remaining() <= 0:
                break
            rnd_idx = random.randint(0, len(self.CL) - 1)
            selected = self.CL.pop(rnd_idx)
            self.sol.add(selected)
        
        while not self.constructive_stop_criteria() and self.CL:
            if self.time_remaining() <= 0:
                break
            self.update_CL()
            if not self.CL:
                break
                
            best_candidate = None
            best_cost = float('-inf')
            
            for c in self.CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost > best_cost:
                    best_cost = delta_cost
                    best_candidate = c
            
            if best_candidate:
                self.CL.remove(best_candidate)
                self.sol.add(best_candidate)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def sampled_greedy_construction(self) -> Solution[E]:
        self.CL = self.make_CL()
        self.sol = self.create_empty_sol()
        sample_size = max(2, int(0.5 * len(self.CL)))
        
        while not self.constructive_stop_criteria() and self.CL:
            if self.time_remaining() <= 0:
                break
            self.update_CL()
            if not self.CL:
                break
            
            sample_size_current = min(sample_size, len(self.CL))
            sampled_candidates = random.sample(self.CL, sample_size_current)
            
            best_candidate = None
            best_cost = float('-inf')
            
            for c in sampled_candidates:
                delta_cost = self.obj_function.evaluate_insertion_cost(c, self.sol)
                if delta_cost > best_cost:
                    best_cost = delta_cost
                    best_candidate = c
            
            if best_candidate:
                self.CL.remove(best_candidate)
                self.sol.add(best_candidate)
                self.obj_function.evaluate(self.sol)
        
        return self.sol
    
    def solve(self) -> Solution[E]:
        self.best_sol = self.create_empty_sol()
        iterations_without_improvement = 0
        
        for i in range(self.iterations):
            if self.time_remaining() <= 0:
                if self.verbose:
                    print(f"\nParada por tempo limite atingido")
                break
            
            self.constructive_heuristic()
            
            if not self.obj_function.is_feasible(self.sol):
                self.make_feasible()
            
            self.local_search()
            
            if self.best_sol.cost < self.sol.cost:
                self.best_sol = Solution(self.sol)
                iterations_without_improvement = 0
                if self.verbose:
                    elapsed = time.time() - self.start_time
                    print(f"(Iter. {i:4d} | Time: {elapsed:6.2f}s) BestSol = {self.best_sol}")
            else:
                iterations_without_improvement += 1
            
            if iterations_without_improvement >= self.max_iter_no_improvement:
                if self.verbose:
                    elapsed = time.time() - self.start_time
                    print(f"\nParada por convergência após {iterations_without_improvement} iterações sem melhoria")
                    print(f"Total de iterações executadas: {i+1}")
                break
        
        return self.best_sol
    
    def constructive_stop_criteria(self) -> bool:
        return self.obj_function.is_feasible(self.sol)


# ============================================================================
# GRASP_QBF_SC
# ============================================================================

class GRASP_QBF_SC(AbstractGRASP[int]):
    """GRASP implementation for QBF with Set Cover constraints"""
    
    def __init__(self, alpha: float, iterations: int, filename: str, 
                 construction_method: str = "standard", 
                 local_search_method: str = "first_improving",
                 time_limit: float = None,
                 max_iter_no_improvement: int = 1000,
                 target_value: float = None,
                 verbose: bool = True):
        super().__init__(QBF_SC(filename), alpha, iterations, construction_method, 
                        local_search_method, max_iter_no_improvement, verbose)
        self.time_limit = time_limit
        self.target_value = target_value
        self.time_to_target = None
        self.iterations_completed = 0
    
    def make_CL(self) -> List[int]:
        return list(range(self.obj_function.get_domain_size()))
    
    def make_RCL(self) -> List[int]:
        return []
    
    def update_CL(self):
        if self.obj_function.is_feasible(self.sol):
            self.CL = [i for i in range(self.obj_function.get_domain_size()) if i not in self.sol]
        else:
            uncovered = self.obj_function.get_uncovered_variables(self.sol)
            self.CL = []
            for i in range(self.obj_function.get_domain_size()):
                if i not in self.sol:
                    subset_coverage = self.obj_function.subsets.get(i, set())
                    if uncovered.intersection(subset_coverage):
                        self.CL.append(i)
    
    def create_empty_sol(self) -> Solution[int]:
        sol = Solution[int]()
        sol.cost = float('-inf')
        return sol
    
    def make_feasible(self):
        max_attempts = 100
        attempts = 0
        
        while not self.obj_function.is_feasible(self.sol) and attempts < max_attempts:
            uncovered = self.obj_function.get_uncovered_variables(self.sol)
            if not uncovered:
                break
            
            best_subset = None
            best_coverage = 0
            
            for i in range(self.obj_function.get_domain_size()):
                if i not in self.sol:
                    subset_coverage = self.obj_function.subsets.get(i, set())
                    coverage_count = len(uncovered.intersection(subset_coverage))
                    if coverage_count > best_coverage:
                        best_coverage = coverage_count
                        best_subset = i
            
            if best_subset is not None:
                self.sol.add(best_subset)
            else:
                break
            
            attempts += 1
        
        self.obj_function.evaluate(self.sol)
    
    def local_search(self) -> Solution[int]:
        improved = True
        
        while improved:
            if self.time_remaining() <= 0:
                break
            
            improved = False
            best_move = None
            best_delta = 0.0
            
            current_CL = [i for i in range(self.obj_function.get_domain_size()) if i not in self.sol]
            
            for cand_in in current_CL:
                delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.sol)
                if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                    best_delta = delta_cost
                    best_move = ("insert", cand_in, None)
                    if self.local_search_method == "first_improving" and delta_cost > 0:
                        break
            
            if not (self.local_search_method == "first_improving" and best_move):
                for cand_out in list(self.sol):
                    delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.sol)
                    if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                        best_delta = delta_cost
                        best_move = ("remove", None, cand_out)
                        if self.local_search_method == "first_improving" and delta_cost > 0:
                            break
            
            if not (self.local_search_method == "first_improving" and best_move):
                for cand_in in current_CL:
                    for cand_out in list(self.sol):
                        delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.sol)
                        if delta_cost > best_delta or (self.local_search_method == "first_improving" and delta_cost > 0):
                            best_delta = delta_cost
                            best_move = ("exchange", cand_in, cand_out)
                            if self.local_search_method == "first_improving" and delta_cost > 0:
                                break
                    if self.local_search_method == "first_improving" and best_move and best_move[0] == "exchange":
                        break
            
            if best_move and best_delta > 1e-10:
                move_type, cand_in, cand_out = best_move
                
                if move_type == "insert" and cand_in is not None:
                    self.sol.add(cand_in)
                elif move_type == "remove" and cand_out is not None:
                    self.sol.remove(cand_out)
                elif move_type == "exchange" and cand_in is not None and cand_out is not None:
                    self.sol.remove(cand_out)
                    self.sol.add(cand_in)
                
                self.obj_function.evaluate(self.sol)
                improved = True
        
        return self.sol
    
    def solve(self) -> Solution[int]:
        self.best_sol = self.create_empty_sol()
        iterations_without_improvement = 0
        
        for i in range(self.iterations):
            self.iterations_completed = i + 1
            
            if self.time_remaining() <= 0:
                if self.verbose:
                    print(f"\nParada por tempo limite atingido")
                break
            
            self.constructive_heuristic()
            
            if not self.obj_function.is_feasible(self.sol):
                self.make_feasible()
            
            self.local_search()
            
            if self.best_sol.cost < self.sol.cost:
                self.best_sol = Solution(self.sol)
                iterations_without_improvement = 0
                
                if self.target_value is not None and self.time_to_target is None:
                    if self.best_sol.cost >= self.target_value:
                        self.time_to_target = time.time() - self.start_time
                        if self.verbose:
                            print(f"\n✓ Alvo {self.target_value} atingido na iteração {i}")
                            print(f"  Tempo até alvo: {self.time_to_target:.2f}s")
                
                if self.verbose:
                    elapsed = time.time() - self.start_time
                    print(f"(Iter. {i:4d} | Time: {elapsed:6.2f}s) BestSol = {self.best_sol}")
            else:
                iterations_without_improvement += 1
            
            if iterations_without_improvement >= self.max_iter_no_improvement:
                if self.verbose:
                    elapsed = time.time() - self.start_time
                    print(f"\nParada por convergência após {iterations_without_improvement} iterações sem melhoria")
                    print(f"Total de iterações executadas: {i+1}")
                break
        
        return self.best_sol


# ============================================================================
# MAIN
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='GRASP para MAX-SC-QBF - Ajustado para TTT-Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
  python grasp_adjusted_ttt.py instances/qbf_sc/instance-01.txt 600 --seed 0 --target 2267 --csv results.csv
  python grasp_adjusted_ttt.py instances/qbf_sc/instance-04.txt 600 --alpha 0.15 --construction random_plus_greedy --seed 5
        """
    )
    
    parser.add_argument('instance', help='Arquivo da instância')
    parser.add_argument('time_limit', type=float, help='Tempo limite em segundos')
    
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Parâmetro alpha para RCL (padrão: 0.1)')
    
    parser.add_argument('--construction', choices=['standard', 'random_plus_greedy', 'sampled_greedy'],
                       default='standard',
                       help='Método de construção (padrão: standard)')
    
    parser.add_argument('--local-search', choices=['first_improving', 'best_improving'],
                       default='first_improving',
                       help='Método de busca local (padrão: first_improving)')
    
    parser.add_argument('--max-iter-no-improvement', type=int, default=1000,
                       help='Máximo de iterações sem melhoria antes de parar (padrão: 1000)')
    
    parser.add_argument('--seed', type=int, default=0,
                       help='Seed para gerador aleatório (padrão: 0)')
    
    parser.add_argument('--target', type=float, default=None,
                       help='Valor alvo para TTT-plot (opcional)')
    
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Exibe informações detalhadas durante execução')
    
    parser.add_argument('--quiet', action='store_true', default=False,
                       help='Modo silencioso, apenas resultado final')
    
    parser.add_argument('--csv', type=str, default=None,
                       help='Arquivo CSV de saída (formato TTT-plot)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    verbose = args.verbose
    if args.quiet:
        verbose = False
    else:
        verbose = True
    
    random.seed(args.seed)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MAX-SC-QBF Solver usando GRASP")
        print(f"{'='*70}")
        print(f"Instância: {args.instance}")
        print(f"Tempo limite: {args.time_limit} segundos")
        print(f"Configuração:")
        print(f"  - Alpha: {args.alpha}")
        print(f"  - Construção: {args.construction}")
        print(f"  - Busca Local: {args.local_search}")
        print(f"  - Max iter. sem melhoria: {args.max_iter_no_improvement}")
        print(f"  - Seed: {args.seed}")
        if args.target:
            print(f"  - Valor alvo: {args.target}")
        print(f"{'='*70}\n")
    
    try:
        iterations = 1000000
        
        grasp = GRASP_QBF_SC(
            alpha=args.alpha,
            iterations=iterations,
            filename=args.instance,
            construction_method=args.construction,
            local_search_method=args.local_search,
            time_limit=args.time_limit,
            max_iter_no_improvement=args.max_iter_no_improvement,
            target_value=args.target,
            verbose=verbose
        )
        
        if verbose:
            print(f"Instância carregada com sucesso!")
            print(f"Número de variáveis: {grasp.obj_function.size}")
            print(f"Número de subconjuntos: {len(grasp.obj_function.subsets)}\n")
        
        start = time.time()
        grasp.start_time = start
        best_solution = grasp.solve()
        end = time.time()
        
        # Extract instance name
        instance_name = os.path.basename(args.instance)
        
        # Build algorithm name with parameters
        algorithm_name = f"GRASP_alpha{args.alpha}"
        if args.construction != 'standard':
            algorithm_name += f"_{args.construction}"
        if args.local_search != 'first_improving':
            algorithm_name += f"_{args.local_search}"
        
        # Determine hit
        hit = 1 if (args.target is not None and grasp.time_to_target is not None) else 0
        
        # Time to target
        time_sec = grasp.time_to_target if grasp.time_to_target is not None else (end - start)
        
        if args.quiet:
            print(f"Custo: {best_solution.cost:.2f}")
            print(f"Tempo: {end - start:.2f}")
            if grasp.time_to_target is not None:
                print(f"TempoAlvo: {grasp.time_to_target:.2f}")
            print(f"Subconjuntos: {len(best_solution)}")
            print(f"Iteracoes: {grasp.iterations_completed}")
        else:
            print(f"\n{'='*70}")
            print("RESULTADO FINAL")
            print(f"{'='*70}")
            print(f"Melhor custo: {best_solution.cost:.2f}")
            print(f"Tamanho da solução: {len(best_solution)}")
            print(f"Viável: {grasp.obj_function.is_feasible(best_solution)}")
            print(f"Tempo de execução: {end - start:.2f} segundos")
            print(f"Iterações executadas: {grasp.iterations_completed}")
            if grasp.time_to_target is not None:
                print(f"Tempo até alvo: {grasp.time_to_target:.2f}s")
            
            if verbose:
                print(f"Subconjuntos selecionados: {sorted(best_solution.elements)}")
            
            print(f"{'='*70}\n")
        
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
                    'best_cost': f'{best_solution.cost:.2f}'
                })
        
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()