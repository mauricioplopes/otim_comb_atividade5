#!/usr/bin/env python3
"""
tabu_search_qbf_sc_ttt.py

Script único para resolver o problema MAX SC QBF usando Tabu Search.
Versão adaptada para gerar saídas compatíveis com ttt-plot.

Uso: python tabu_search_qbf_sc_ttt.py <tenure> <iterations> <filename> [options]

Parâmetros:
  tenure      : Tamanho da lista tabu
  iterations  : Número de iterações
  filename    : Arquivo da instância MAX SC QBF
  
Opções:
  debug       : Ativa modo debug detalhado
  quiet       : Desativa saídas verbosas
  seed=N      : Define seed aleatória (padrão: 0)
  target=N    : Define valor alvo para TTT (padrão: calcula automaticamente)
  timeout=N   : Define timeout em segundos (padrão: 600)
  output=file : Define arquivo CSV de saída (padrão: results.csv)
  
Nota: O script sempre faz append ao CSV se ele já existir, ou cria novo se não existir.
"""

import sys
import time
import random
import traceback
import csv
import os
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
from typing import List, Any, Optional


# ============================================================================
# CLASSE SOLUTION
# ============================================================================

class Solution(list):
    """
    Classe que representa uma solução para problemas de otimização.
    Herda de list para permitir operações como append, remove, etc.
    """
    
    def __init__(self, solution=None):
        super().__init__()
        self.cost = float('inf')
        
        if solution is not None:
            self.extend(solution)
            self.cost = solution.cost if hasattr(solution, 'cost') else float('inf')
    
    def copy(self):
        return Solution(self)
    
    def is_empty(self):
        return len(self) == 0
    
    def get_cost(self):
        return self.cost
    
    def set_cost(self, cost):
        self.cost = cost
    
    def get_size(self):
        return len(self)
    
    def contains_element(self, element):
        return element in self
    
    def add_element(self, element):
        if element not in self:
            self.append(element)
    
    def remove_element(self, element):
        if element in self:
            self.remove(element)
            return True
        return False
    
    def get_elements(self):
        return list(self)
    
    def __str__(self):
        return f"Solution: cost={self.cost:.6f}, size={len(self)}, elements={list(self)}"


# ============================================================================
# INTERFACE EVALUATOR
# ============================================================================

class Evaluator(ABC):
    """Interface abstrata para avaliadores de função objetivo."""
    
    @abstractmethod
    def get_domain_size(self) -> int:
        pass
    
    @abstractmethod
    def evaluate(self, solution: Solution) -> float:
        pass
    
    @abstractmethod
    def evaluate_insertion_cost(self, element: Any, solution: Solution) -> float:
        pass
    
    @abstractmethod
    def evaluate_removal_cost(self, element: Any, solution: Solution) -> float:
        pass
    
    @abstractmethod
    def evaluate_exchange_cost(self, element_in: Any, element_out: Any, solution: Solution) -> float:
        pass


# ============================================================================
# CLASSE QBF
# ============================================================================

class QBF(Evaluator):
    """
    Classe para o problema Quadratic Binary Function.
    Implementa f(x) = x^T * A * x onde x é um vetor binário.
    """
    
    def __init__(self, filename: str):
        self.A = None
        self.size = self._read_input(filename)
        self.variables = [1.0] * self.size
    
    def _read_input(self, filename: str) -> int:
        """Lê o arquivo de entrada e inicializa a matriz A."""
        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            if not lines:
                raise ValueError("Arquivo vazio")
            
            n = int(lines[0])
            
            self.A = [[0.0] * n for _ in range(n)]
            
            if len(lines) < n + 1:
                return n
            
            line_idx = 1
            for i in range(n):
                if line_idx >= len(lines):
                    break
                
                try:
                    values = list(map(float, lines[line_idx].split()))
                    
                    for j, val in enumerate(values):
                        col_idx = i + j
                        if col_idx < n:
                            self.A[i][col_idx] = val
                            if col_idx != i:
                                self.A[col_idx][i] = 0.0
                
                except ValueError as e:
                    pass
                
                line_idx += 1
            
            return n
            
        except FileNotFoundError:
            print(f"ERRO: Arquivo '{filename}' não encontrado!")
            self.A = [[0.0]]
            return 1
        except Exception as e:
            print(f"ERRO ao ler arquivo {filename}: {e}")
            self.A = [[0.0]]
            return 1
    
    def reset_variables(self):
        self.variables = [1.0] * self.size
    
    def set_variables(self, solution: Solution):
        """Define as variáveis baseado na solução."""
        self.reset_variables()
        if solution:
            for elem in solution:
                if 0 <= elem < self.size:
                    self.variables[elem] = 0.0
    
    def get_domain_size(self) -> int:
        return self.size
    
    def evaluate(self, solution: Solution) -> float:
        self.set_variables(solution)
        solution.cost = self._evaluate_qbf()
        return solution.cost
    
    def _evaluate_qbf(self) -> float:
        """Calcula f(x) = x^T * A * x."""
        total = 0.0
        vec_aux = [0.0] * self.size
        
        for i in range(self.size):
            aux = 0.0
            for j in range(self.size):
                aux += self.variables[j] * self.A[i][j]
            vec_aux[i] = aux
            total += aux * self.variables[i]
        
        return total
    
    def evaluate_insertion_cost(self, elem: int, solution: Solution) -> float:
        self.set_variables(solution)
        return self._evaluate_removal_qbf(elem)
    
    def _evaluate_insertion_qbf(self, i: int) -> float:
        if self.variables[i] == 1:
            return 0.0
        return self._evaluate_contribution_qbf(i)
    
    def evaluate_removal_cost(self, elem: int, solution: Solution) -> float:
        self.set_variables(solution)
        return self._evaluate_insertion_qbf(elem)
    
    def _evaluate_removal_qbf(self, i: int) -> float:
        if self.variables[i] == 0:
            return 0.0
        return -self._evaluate_contribution_qbf(i)
    
    def evaluate_exchange_cost(self, elem_in: int, elem_out: int, solution: Solution) -> float:
        self.set_variables(solution)
        return self._evaluate_exchange_qbf(elem_out, elem_in)
    
    def _evaluate_exchange_qbf(self, elem_in: int, elem_out: int) -> float:
        if elem_in == elem_out:
            return 0.0
        if self.variables[elem_in] == 1:
            return self._evaluate_removal_qbf(elem_out)
        if self.variables[elem_out] == 0:
            return self._evaluate_insertion_qbf(elem_in)
        
        total = 0.0
        total += self._evaluate_contribution_qbf(elem_in)
        total -= self._evaluate_contribution_qbf(elem_out)
        total -= (self.A[elem_in][elem_out] + self.A[elem_out][elem_in])
        
        return total
    
    def _evaluate_contribution_qbf(self, i: int) -> float:
        if self.A is None:
            return 0.0
        
        if not (0 <= i < self.size):
            return 0.0
        
        total = 0.0
        
        for j in range(self.size):
            if i != j:
                try:
                    total += self.variables[j] * (self.A[i][j] + self.A[j][i])
                except (IndexError, TypeError) as e:
                    return 0.0
        
        try:
            total += self.A[i][i]
        except (IndexError, TypeError) as e:
            return 0.0
        
        return total


# ============================================================================
# CLASSE QBFINVERSE
# ============================================================================

class QBFInverse(QBF):
    """Versão inversa da QBF para uso em algoritmos de minimização."""
    
    def _evaluate_qbf(self) -> float:
        return -super()._evaluate_qbf()
    
    def _evaluate_insertion_qbf(self, i: int) -> float:
        return -super()._evaluate_insertion_qbf(i)
    
    def _evaluate_removal_qbf(self, i: int) -> float:
        return -super()._evaluate_removal_qbf(i)
    
    def _evaluate_exchange_qbf(self, elem_in: int, elem_out: int) -> float:
        return -super()._evaluate_exchange_qbf(elem_in, elem_out)


# ============================================================================
# CLASSE QBFSCINVERSE - MAX SC QBF
# ============================================================================

class QBFSCInverse(QBFInverse):
    """Implementação do QBF com restrição de Set Cover."""
    
    def __init__(self, filename: str):
        self.sets = []
        super().__init__(filename)

    def _read_input(self, filename: str) -> int:
        """Lê o arquivo de entrada para MAX SC QBF."""
        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file.readlines() if line.strip()]
            
            if not lines:
                raise ValueError("Arquivo vazio")
            
            n = int(lines[0])
            
            self.A = [[0.0] * n for _ in range(n)]

            self.sets = []
            
            if len(lines) < 2 * (n + 1):
                return n
            
            line_idx = 2
            for i in range(n):
                if line_idx >= len(lines):
                    break
                
                try:
                    self.sets.append(set(map(int, lines[line_idx].split())))
                except ValueError as e:
                    raise ValueError(f"Erro ao processar linha {line_idx}: {e}") from e
                
                line_idx += 1
            
            for i in range(n):
                if line_idx >= len(lines):
                    break
                
                try:
                    values = list(map(float, lines[line_idx].split()))
                    
                    for j, val in enumerate(values):
                        col_idx = i + j
                        if col_idx < n:
                            self.A[i][col_idx] = val
                            if col_idx != i:
                                self.A[col_idx][i] = 0.0
                
                except ValueError as e:
                    raise ValueError(f"Erro ao processar linha {line_idx}: {e}") from e
                
                line_idx += 1
            
            return n
            
        except FileNotFoundError:
            print(f"ERRO: Arquivo '{filename}' não encontrado!")
            self.A = [[0.0]]
            return 1
        except Exception as e:
            print(f"ERRO ao ler arquivo {filename}: {e}")
            self.A = [[0.0]]
            return 1
        
    def get_variables_that_can_be_set_to_zero(self) -> List[int]:
        """Retorna variáveis que podem ser definidas como 0 sem violar a restrição de cobertura."""
        set_indexes_enabled = [i for i in range(self.size) if self.variables[i] == 1.0]

        element_coverage_count = {}
        for set_index in set_indexes_enabled:
            for elem in self.sets[set_index]:
                element_coverage_count[elem] = element_coverage_count.get(elem, 0) + 1

        variables_that_can_be_set_to_zero = []
        for i in range(self.size):
            if self.variables[i] == 1.0:
                can_be_set_to_zero = True
                for elem in self.sets[i]:
                    if element_coverage_count.get(elem, 0) == 1:
                        can_be_set_to_zero = False
                        break
                if can_be_set_to_zero:
                    variables_that_can_be_set_to_zero.append(i)

        return variables_that_can_be_set_to_zero


# ============================================================================
# CLASSE ABSTRACTTABUSEARCH
# ============================================================================

class AbstractTabuSearch(ABC):
    """Classe abstrata para implementação do algoritmo Tabu Search."""
    
    VERBOSE = True
    
    def __init__(self, obj_function: Evaluator, tenure: int, iterations: int, random_seed: int = 0):
        self.obj_function = obj_function
        self.tenure = tenure
        self.iterations = iterations
        
        self.best_sol: Optional[Solution] = None
        self.current_sol: Optional[Solution] = None
        self.best_cost: Optional[float] = None
        self.current_cost: Optional[float] = None
        
        self.candidate_list: Optional[List[Any]] = None
        self.restricted_candidate_list: Optional[List[Any]] = None
        self.tabu_list: Optional[deque] = None
        
        self.fake_element = self._get_fake_element()
        
        random.seed(random_seed)
    
    @abstractmethod
    def _get_fake_element(self) -> Any:
        pass
    
    @abstractmethod
    def make_candidate_list(self) -> List[Any]:
        pass
    
    @abstractmethod
    def make_restricted_candidate_list(self) -> List[Any]:
        pass
    
    @abstractmethod
    def make_tabu_list(self) -> deque:
        pass
    
    @abstractmethod
    def update_candidate_list(self):
        pass
    
    @abstractmethod
    def create_empty_solution(self) -> Solution:
        pass
    
    @abstractmethod
    def neighborhood_move(self) -> Optional[Solution]:
        pass
    
    def constructive_heuristic(self) -> Solution:
        """Heurística construtiva para gerar solução inicial."""
        self.candidate_list = self.make_candidate_list()
        self.restricted_candidate_list = self.make_restricted_candidate_list()
        self.current_sol = self.create_empty_solution()
        
        self.obj_function.evaluate(self.current_sol)
        
        max_iterations = len(self.candidate_list) * 2
        iteration_count = 0
        
        while iteration_count < max_iterations:
            previous_cost = self.current_sol.cost
            self.update_candidate_list()
            
            available_candidates = self._get_available_candidates()
            
            if not available_candidates:
                break
            
            min_cost = float('inf')
            
            candidate_costs = {}
            for candidate in available_candidates:
                delta_cost = self.obj_function.evaluate_insertion_cost(candidate, self.current_sol)
                candidate_costs[candidate] = delta_cost
                if delta_cost < min_cost:
                    min_cost = delta_cost
            
            self.restricted_candidate_list.clear()
            for candidate, delta_cost in candidate_costs.items():
                if delta_cost <= min_cost:
                    self.restricted_candidate_list.append(candidate)
            
            if not self.restricted_candidate_list:
                break
            
            selected_idx = random.randint(0, len(self.restricted_candidate_list) - 1)
            selected_candidate = self.restricted_candidate_list[selected_idx]
            
            self.current_sol.add_element(selected_candidate)
            self.obj_function.evaluate(self.current_sol)
            
            if abs(self.current_sol.cost - previous_cost) < 1e-10:
                break
            
            if self.current_sol.cost > previous_cost + abs(previous_cost) * 0.1:
                break
            
            iteration_count += 1
        
        return self.current_sol
    
    def _get_available_candidates(self) -> List[Any]:
        return [c for c in self.candidate_list if not self.current_sol.contains_element(c)]
    
    def solve(self) -> Solution:
        """Método principal do Tabu Search."""
        self.best_sol = self.create_empty_solution()
        
        self.constructive_heuristic()
        
        self.tabu_list = self.make_tabu_list()
        
        self.best_sol = self.current_sol.copy()
        
        for iteration in range(self.iterations):
            self.neighborhood_move()
            
            if self.current_sol.cost < self.best_sol.cost:
                self.best_sol = self.current_sol.copy()
        
        return self.best_sol
    
    def get_best_solution(self) -> Optional[Solution]:
        return self.best_sol
    
    def get_current_solution(self) -> Optional[Solution]:
        return self.current_sol
    
    def get_tabu_list(self) -> Optional[deque]:
        return self.tabu_list
    
    def is_tabu(self, element: Any) -> bool:
        return self.tabu_list is not None and element in self.tabu_list
    
    def aspiration_criteria(self, element: Any, delta_cost: float) -> bool:
        if self.current_sol is None or self.best_sol is None:
            return False
        
        return self.current_sol.cost + delta_cost < self.best_sol.cost
    
    def add_to_tabu_list(self, element: Any):
        if self.tabu_list is not None:
            self.tabu_list.popleft()
            self.tabu_list.append(element)
    
    def set_verbose(self, verbose: bool):
        self.VERBOSE = verbose


# ============================================================================
# CLASSE TABUSEARCHQBFSC
# ============================================================================

class TabuSearchQBFSc(AbstractTabuSearch):
    """Implementação do Tabu Search especializada para o problema QBF com Set Cover."""
    
    def __init__(self, tenure: int, iterations: int, filename: str, random_seed: int = 0):
        qbf_inverse = QBFSCInverse(filename)
        super().__init__(qbf_inverse, tenure, iterations, random_seed)
        self.qbf: QBFSCInverse = qbf_inverse
    
    def _get_fake_element(self) -> int:
        return -1
    
    def make_candidate_list(self) -> List[int]:
        return list(range(self.obj_function.get_domain_size()))
    
    def make_restricted_candidate_list(self) -> List[int]:
        return []
    
    def make_tabu_list(self) -> deque:
        tabu_list = deque(maxlen=2 * self.tenure)
        for _ in range(2 * self.tenure):
            tabu_list.append(self.fake_element)
        return tabu_list
    
    def update_candidate_list(self):
        """Atualiza lista de candidatos considerando restrições de set cover."""
        self.candidate_list = self.qbf.get_variables_that_can_be_set_to_zero()
    
    def create_empty_solution(self) -> Solution:
        solution = Solution()
        solution.cost = self.obj_function.evaluate(solution)
        return solution
    
    def neighborhood_move(self) -> Optional[Solution]:
        """Executa movimento de vizinhança explorando inserção, remoção e troca."""
        min_delta_cost = float('inf')
        best_cand_in = None
        best_cand_out = None
        best_move_type = None
        
        self.update_candidate_list()
        
        # 1. AVALIA INSERÇÕES
        for cand_in in self.candidate_list:
            if not self.current_sol.contains_element(cand_in):
                delta_cost = self.obj_function.evaluate_insertion_cost(cand_in, self.current_sol)
                
                if (not self.is_tabu(cand_in) or 
                    self.aspiration_criteria(cand_in, delta_cost)):
                    
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_cand_in = cand_in
                        best_cand_out = None
                        best_move_type = "insertion"
        
        # 2. AVALIA REMOÇÕES
        for cand_out in list(self.current_sol):
            delta_cost = self.obj_function.evaluate_removal_cost(cand_out, self.current_sol)
            
            if (not self.is_tabu(cand_out) or 
                self.aspiration_criteria(cand_out, delta_cost)):
                
                if delta_cost < min_delta_cost:
                    min_delta_cost = delta_cost
                    best_cand_in = None
                    best_cand_out = cand_out
                    best_move_type = "removal"
        
        # 3. AVALIA TROCAS (2-EXCHANGE)
        for cand_in in self.candidate_list:
            if not self.current_sol.contains_element(cand_in):
                for cand_out in list(self.current_sol):
                    delta_cost = self.obj_function.evaluate_exchange_cost(cand_in, cand_out, self.current_sol)
                    
                    tabu_in = self.is_tabu(cand_in)
                    tabu_out = self.is_tabu(cand_out)
                    aspiration = self.aspiration_criteria(cand_in, delta_cost)
                    
                    if (not tabu_in and not tabu_out) or aspiration:
                        if delta_cost < min_delta_cost:
                            min_delta_cost = delta_cost
                            best_cand_in = cand_in
                            best_cand_out = cand_out
                            best_move_type = "exchange"
        
        # 4. IMPLEMENTA O MELHOR MOVIMENTO
        if best_move_type is None:
            return None
        
        if best_move_type == "insertion":
            self.add_to_tabu_list(self.fake_element)
            self.add_to_tabu_list(best_cand_in)
            self.current_sol.add_element(best_cand_in)
            
        elif best_move_type == "removal":
            self.add_to_tabu_list(best_cand_out)
            self.add_to_tabu_list(self.fake_element)
            self.current_sol.remove_element(best_cand_out)
            
        elif best_move_type == "exchange":
            self.add_to_tabu_list(best_cand_out)
            self.add_to_tabu_list(best_cand_in)
            self.current_sol.remove_element(best_cand_out)
            self.current_sol.add_element(best_cand_in)
        
        self.obj_function.evaluate(self.current_sol)
        
        return None


# ============================================================================
# CLASSE TTTTRACKER - RASTREAMENTO PARA TTT-PLOT
# ============================================================================

class TTTTracker:
    """Rastreia o progresso do algoritmo para gerar saídas TTT-plot."""
    
    def __init__(self, instance_name: str, algorithm_name: str, seed: int, target_value: float, timeout: float):
        self.instance_name = instance_name
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.target_value = target_value
        self.timeout = timeout
        
        self.start_time = None
        self.target_reached = False
        self.time_to_target = None
        self.best_cost_found = float('-inf')
        self.finished = False
    
    def start(self):
        """Inicia o rastreamento."""
        self.start_time = time.time()
    
    def update(self, current_best_cost: float):
        """Atualiza o estado com o melhor custo atual."""
        # Lembre-se: no código, trabalhamos com custo invertido (minimização)
        # Então precisamos inverter novamente para obter o valor real
        real_value = -current_best_cost
        
        if real_value > self.best_cost_found:
            self.best_cost_found = real_value
        
        # Verifica se atingiu o target
        if not self.target_reached and real_value >= self.target_value:
            self.target_reached = True
            self.time_to_target = time.time() - self.start_time
    
    def finish(self):
        """Finaliza o rastreamento."""
        self.finished = True
        if not self.target_reached:
            # Se não atingiu o target, o tempo é o timeout
            self.time_to_target = time.time() - self.start_time
    
    def get_result(self) -> dict:
        """Retorna o resultado no formato do CSV."""
        return {
            'instance': self.instance_name,
            'target': self.target_value,
            'algorithm': self.algorithm_name,
            'seed': self.seed,
            'time_sec': self.time_to_target if self.time_to_target else (time.time() - self.start_time),
            'hit': 1 if self.target_reached else 0,
            'best_cost': self.best_cost_found
        }


# ============================================================================
# CLASSE TABUSEARCHQBFSC COM TTT
# ============================================================================

class TabuSearchQBFScTTT(TabuSearchQBFSc):
    """Versão do Tabu Search com rastreamento para TTT-plot."""
    
    def __init__(self, tenure: int, iterations: int, filename: str, random_seed: int = 0, 
                 ttt_tracker: Optional[TTTTracker] = None):
        super().__init__(tenure, iterations, filename, random_seed)
        self.ttt_tracker = ttt_tracker
    
    def solve(self) -> Solution:
        """Método principal do Tabu Search com rastreamento TTT."""
        if self.ttt_tracker:
            self.ttt_tracker.start()
        
        self.best_sol = self.create_empty_solution()
        
        self.constructive_heuristic()
        
        self.tabu_list = self.make_tabu_list()
        
        self.best_sol = self.current_sol.copy()
        
        # Atualiza após construção
        if self.ttt_tracker:
            self.ttt_tracker.update(self.best_sol.cost)
        
        for iteration in range(self.iterations):
            self.neighborhood_move()
            
            if self.current_sol.cost < self.best_sol.cost:
                self.best_sol = self.current_sol.copy()
                
                # Atualiza o tracker
                if self.ttt_tracker:
                    self.ttt_tracker.update(self.best_sol.cost)
                    
                    # Se já atingiu o target, pode parar
                    if self.ttt_tracker.target_reached:
                        if self.VERBOSE:
                            print(f"Target atingido na iteração {iteration}!")
                        break
        
        if self.ttt_tracker:
            self.ttt_tracker.finish()
        
        return self.best_sol


# ============================================================================
# FUNÇÕES AUXILIARES DO PROGRAMA PRINCIPAL
# ============================================================================

def print_usage():
    """Imprime instruções de uso do programa."""
    print("Uso: python tabu_search_qbf_sc_ttt.py <tenure> <iterations> <filename> [options]")
    print()
    print("Parâmetros obrigatórios:")
    print("  tenure      : Tamanho da lista tabu (ex: 20)")
    print("  iterations  : Número de iterações (ex: 1000)")
    print("  filename    : Arquivo da instância MAX SC QBF")
    print()
    print("Opções:")
    print("  debug       : Ativa modo debug detalhado")
    print("  quiet       : Desativa saídas verbosas")
    print("  seed=N      : Define seed aleatória (ex: seed=42)")
    print("  target=N    : Define valor alvo para TTT (ex: target=5000)")
    print("  timeout=N   : Define timeout em segundos (padrão: 600)")
    print("  output=file : Define arquivo CSV de saída (padrão: results.csv)")
    print()
    print("Nota: O script sempre faz append ao CSV se ele já existir.")
    print()
    print("Exemplos:")
    print("  python tabu_search_qbf_sc_ttt.py 20 1000 instances/qbf200")
    print("  python tabu_search_qbf_sc_ttt.py 20 1000 instances/qbf200 target=5000 seed=42")
    print("  python tabu_search_qbf_sc_ttt.py 20 1000 instances/qbf200 output=myresults.csv")


def parse_arguments(args):
    """Analisa os argumentos da linha de comando."""
    if len(args) < 4:
        return None
    
    try:
        parsed = {
            'tenure': int(args[1]),
            'iterations': int(args[2]),
            'filename': args[3],
            'debug': False,
            'quiet': False,
            'seed': 0,
            'target': None,
            'timeout': 600.0,
            'output': 'results.csv'
        }
        
        for arg in args[4:]:
            arg_lower = arg.lower()
            
            if arg_lower == 'debug':
                parsed['debug'] = True
            elif arg_lower == 'quiet':
                parsed['quiet'] = True
            elif arg_lower.startswith('seed='):
                try:
                    parsed['seed'] = int(arg_lower.split('=')[1])
                except (ValueError, IndexError):
                    print(f"AVISO: Seed inválida '{arg}', usando seed=0")
            elif arg_lower.startswith('target='):
                try:
                    parsed['target'] = float(arg_lower.split('=')[1])
                except (ValueError, IndexError):
                    print(f"AVISO: Target inválido '{arg}', será calculado automaticamente")
            elif arg_lower.startswith('timeout='):
                try:
                    parsed['timeout'] = float(arg_lower.split('=')[1])
                except (ValueError, IndexError):
                    print(f"AVISO: Timeout inválido '{arg}', usando timeout=600")
            elif arg_lower.startswith('output='):
                try:
                    parsed['output'] = arg.split('=')[1]
                except IndexError:
                    print(f"AVISO: Output inválido '{arg}', usando output=results.csv")
            else:
                print(f"AVISO: Opção desconhecida '{arg}' ignorada")
        
        return parsed
        
    except ValueError as e:
        print(f"Erro nos parâmetros: {e}")
        return None


def get_instance_name(filename: str) -> str:
    """Extrai o nome da instância do caminho do arquivo."""
    return Path(filename).stem


def calculate_auto_target(qbf: QBFSCInverse, percentage: float = 0.8) -> float:
    """
    Calcula um valor alvo automático baseado em uma estimativa.
    Usa uma porcentagem da diagonal da matriz como estimativa.
    """
    try:
        diagonal_sum = sum(qbf.A[i][i] for i in range(qbf.size))
        estimated_max = diagonal_sum * percentage
        return estimated_max
    except:
        return 0.0


def write_csv_result(filename: str, result: dict):
    """
    Escreve o resultado no arquivo CSV.
    Se o arquivo existir, faz append. Se não existir, cria novo com header.
    """
    file_exists = os.path.exists(filename)
    mode = 'a' if file_exists else 'w'
    
    with open(filename, mode, newline='') as csvfile:
        fieldnames = ['instance', 'target', 'algorithm', 'seed', 'time_sec', 'hit', 'best_cost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Escreve header apenas se for arquivo novo
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(result)


def validate_parameters(params):
    """Valida os parâmetros fornecidos."""
    errors = []
    
    if params['tenure'] <= 0:
        errors.append("Tenure deve ser maior que zero")
    
    if params['iterations'] <= 0:
        errors.append("Iterations deve ser maior que zero")
    
    if params['timeout'] <= 0:
        errors.append("Timeout deve ser maior que zero")
    
    if params['tenure'] > 100:
        print(f"AVISO: Tenure muito alto ({params['tenure']}), pode impactar performance")
    
    if params['iterations'] > 10000:
        print(f"AVISO: Muitas iterações ({params['iterations']}), execução pode ser lenta")
    
    if errors:
        print("Erros de validação:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def run_tabu_search_ttt(params):
    """Executa o Tabu Search com rastreamento TTT."""
    instance_name = get_instance_name(params['filename'])
    
    # Cria o nome do algoritmo com os parâmetros
    algorithm_name = f"TS_t{params['tenure']}_i{params['iterations']}"
    
    if not params['quiet']:
        print("=== TABU SEARCH PARA MAX SC QBF (TTT Mode) ===")
        print(f"Arquivo: {params['filename']}")
        print(f"Instância: {instance_name}")
        print(f"Tenure: {params['tenure']}")
        print(f"Iterations: {params['iterations']}")
        print(f"Seed: {params['seed']}")
        print(f"Timeout: {params['timeout']}s")
        print(f"Output: {params['output']} (append automático)")
        if params['debug']:
            print("Modo DEBUG ativado")
        print()
    
    try:
        # Primeiro, carrega a instância para calcular target se necessário
        qbf_temp = QBFSCInverse(params['filename'])
        
        target_value = params['target']
        if target_value is None:
            target_value = calculate_auto_target(qbf_temp, percentage=0.8)
            if not params['quiet']:
                print(f"Target calculado automaticamente: {target_value:.2f}")
        
        # Cria o tracker TTT
        ttt_tracker = TTTTracker(
            instance_name=instance_name,
            algorithm_name=algorithm_name,
            seed=params['seed'],
            target_value=target_value,
            timeout=params['timeout']
        )
        
        # Cria e executa o Tabu Search
        ts = TabuSearchQBFScTTT(
            tenure=params['tenure'],
            iterations=params['iterations'],
            filename=params['filename'],
            random_seed=params['seed'],
            ttt_tracker=ttt_tracker
        )
        
        ts.set_verbose(not params['quiet'])
        
        if params['debug'] and not params['quiet']:
            print("Instância carregada com sucesso!")
            print(f"Número de conjuntos: {len(ts.qbf.sets)}")
            print()
        
        start_time = time.time()
        
        # Executa com timeout
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Tempo limite excedido")
        
        # Configura timeout (apenas em sistemas Unix)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(params['timeout']))
        except:
            pass  # Windows não suporta SIGALRM
        
        try:
            best_sol = ts.solve()
        except TimeoutError:
            if not params['quiet']:
                print("Timeout atingido!")
            best_sol = ts.get_best_solution()
        finally:
            try:
                signal.alarm(0)  # Cancela o alarme
            except:
                pass
        
        exec_time = time.time() - start_time
        
        # Obtém resultado do tracker
        result = ttt_tracker.get_result()
        
        # Escreve no CSV (sempre faz append se arquivo existir)
        write_csv_result(params['output'], result)
        
        return {
            'success': True,
            'best_solution': best_sol,
            'execution_time': exec_time,
            'result': result,
            'tabu_search': ts
        }
        
    except MemoryError:
        return {
            'success': False,
            'error': "Erro de memória. Instância muito grande.",
            'traceback': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Erro durante execução: {e}",
            'traceback': traceback.format_exc() if params['debug'] else None
        }


def print_results(results, params):
    """Imprime os resultados da execução."""
    if not results['success']:
        print(f"ERRO: {results['error']}")
        if results.get('traceback') and params['debug']:
            print("\nTraceback:")
            print(results['traceback'])
        return
    
    result = results['result']
    exec_time = results['execution_time']
    
    if not params['quiet']:
        print("\n" + "="*50)
        print("RESULTADOS FINAIS")
        print("="*50)
        
        print(f"Instância: {result['instance']}")
        print(f"Algoritmo: {result['algorithm']}")
        print(f"Seed: {result['seed']}")
        print(f"Target: {result['target']:.6f}")
        print(f"Melhor custo encontrado: {result['best_cost']:.6f}")
        print(f"Hit: {result['hit']} ({'Sim' if result['hit'] else 'Não'})")
        print(f"Tempo para target: {result['time_sec']:.3f}s")
        print(f"Tempo total: {exec_time:.3f}s")
        
        print(f"\nResultado salvo em: {params['output']}")
        print("="*50)
    
    else:
        # Modo quiet: apenas resultado essencial
        print(f"{result['algorithm']},{result['seed']},{result['hit']},{result['time_sec']:.3f},{result['best_cost']:.6f}")


# ============================================================================
# FUNÇÃO MAIN
# ============================================================================

def main():
    """Função principal do programa."""
    params = parse_arguments(sys.argv)
    
    if params is None:
        print_usage()
        sys.exit(1)
    
    if not validate_parameters(params):
        sys.exit(1)
    
    results = run_tabu_search_ttt(params)
    
    print_results(results, params)
    
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
