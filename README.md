# Otimização Combinatória - Atividade 5

Análise comparativa de algoritmos de otimização combinatória aplicados ao problema **MAX-SC-QBF** (Maximização de Função Binária Quadrática com Set Cover).

## 📋 Sobre o Problema

O **MAX-SC-QBF** é um problema NP-difícil que combina:
- **Maximização QBF**: otimizar uma função objetivo binária quadrática
- **Set Cover**: satisfazer restrições de cobertura de conjuntos

Este tipo de problema aparece em aplicações práticas como alocação de recursos, design de redes e problemas de decisão em sistemas combinatórios.

## 🔬 Algoritmos Avaliados

O repositório compara **4 abordagens**:

1. **PLI** - Programação Linear Inteira (método exato)
2. **GRASP** - Greedy Randomized Adaptive Search Procedure
3. **Tabu Search** - Busca Tabu
4. **AG** - Algoritmo Genético

Cada meta-heurística foi testada com diferentes configurações de parâmetros e operadores para avaliar o impacto no desempenho.

## 📊 Metodologia de Avaliação

### Performance Profile
Técnica proposta por Dolan e Moré para comparar algoritmos através da função de distribuição cumulativa da razão de desempenho. Permite identificar:
- **ρ(1)**: eficiência (% de vezes que o algoritmo foi o melhor)
- **ρ(τ)**: robustez (% de problemas resolvidos dentro de um fator τ do melhor)

### TTT-Plot (Time-to-Target)
Técnica de Aiex et al. que modela o tempo para atingir uma solução alvo usando distribuição exponencial deslocada. Útil para algoritmos estocásticos, permitindo:
- Comparação de convergência entre algoritmos
- Estimativa probabilística de tempo necessário
- Análise de estabilidade através de múltiplas execuções (50 seeds)


## 🚀 Como Usar

### Gerar Performance Profiles
```bash
python3 performance_profile.py
```
Gera 8 gráficos comparativos salvos como PNG.

### Executar Experimentos TTT
```bash
# GRASP
./run_grasp.sh instancia.txt 5000 results/grasp.csv

# Algoritmo Genético  
./run_ga.sh instancia.txt 5000 results/ga.csv

# Tabu Search
./run_tabu.sh instancia.txt 5000 results/tabu.csv
```
Cada script executa 50 seeds com limite de 600 segundos.

### Gerar TTT Plots
```bash
python3 ttt.py results.csv --out ttt_out/ --time-limit 600 --fit \
  --title-prefix "MAX-SC-QBF"
```

## 📈 Conjunto de Teste

- **15 instâncias** (inst_1 a inst_15)
- **50 execuções** por configuração (seeds 0-49)
- **Limite de tempo**: 600 segundos por execução


## 📚 Referências

- **Performance Profile**: Dolan & Moré (2002) - Mathematical Programming
- **TTT Plots**: Aiex, Resende & Ribeiro (2007) - Optimization Letters

---

**Disciplina: Otimização Combinatória**
**Atividade 5**