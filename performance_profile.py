"""
Script Simplificado para Gerar Performance Profile
Incluindo PLI da Atividade 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# DADOS DAS ATIVIDADES 1, 2, 3 E 4
# ============================================================================

instancias = [f'inst_{i}' for i in range(1, 16)]

# PLI (Atividade 1)
pli = [354, 200, 283, 604, 677, 514, 1475, 2265, 2008, 5183, 4315, 4878, 6844, 8922, 8279]

# GRASP (6 variações)
grasp_std = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4780, 5129, 13681, 14917, 14590]
grasp_std_alpha = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4780, 5129, 13681, 14924, 14590]
grasp_std_best = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4782, 5129, 13657, 14786, 14544]
grasp_std_hc1 = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4782, 5129, 13689, 14919, 14589]
grasp_std_hc2 = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4782, 5129, 13690, 14922, 14590]
grasp_std_hc3 = [354, 200, 283, 604, 677, 514, 1531, 2267, 2008, 5371, 4780, 5129, 13681, 14917, 14590]

# Tabu Search (5 variações)
tabu_padrao_fi = [354, 200, 281, 604, 677, 514, 1531, 2265, 2008, 5183, 4315, 4878, 14378, 10550, 13797]
tabu_best_improving = [354, 200, 281, 604, 677, 514, 1531, 2267, 2008, 5304, 4495, 4828, 14419, 10751, 14419]
tabu_padrao_t2 = [354, 200, 281, 604, 677, 514, 1531, 2265, 2008, 5183, 4458, 4878, 14378, 10743, 14005]
tabu_prob_ts = [354, 200, 283, 604, 677, 486, 1531, 2265, 2008, 5183, 4458, 4849, 1, 10743, 14005]
tabu_intensif = [354, 198, 267, 604, 651, 500, 1531, 2265, 2008, 1924, 3894, 4674, 14381, 8524, 7579]

# Algoritmo Genético (5 variações)
ag_padrao = [354, 200, 283, 604, 656, 514, 1513, 2265, 2008, 5304, 4495, 4828, 9167, 10751, 10052]
ag_padrao_pop = [354, 200, 283, 604, 656, 514, 1523, 2267, 2008, 5236, 4501, 4798, 7824, 8868, 8865]
ag_padrao_mut = [354, 200, 283, 604, 677, 514, 1364, 2138, 1853, 3144, 2931, 3257, 5014, 4950, 4272]
ag_padrao_evol1 = [354, 200, 283, 604, 677, 514, 1417, 2265, 2008, 5211, 4554, 4837, 8866, 9968, 9526]
ag_padrao_evol2 = [354, 200, 283, 582, 677, 514, 1047, 1848, 1477, 2305, 1990, 2350, 3152, 4202, 3118]

# ============================================================================
# FUNÇÃO PARA CALCULAR E PLOTAR PERFORMANCE PROFILE
# ============================================================================

def performance_profile(results_df, tau_max=10, log_scale=False, titulo="Performance Profile", arquivo=None):
    """
    Gera Performance Profile
    
    Parâmetros:
    - results_df: DataFrame com instâncias nas linhas e métodos nas colunas
    - tau_max: valor máximo de tau
    - log_scale: True para escala logarítmica
    - titulo: título do gráfico
    - arquivo: nome do arquivo para salvar (opcional)
    """
    
    metodos = results_df.columns
    instancias_local = results_df.index
    n_problemas = len(instancias_local)
    
    # Calcula ratios
    ratios = pd.DataFrame(index=instancias_local, columns=metodos)
    for inst in instancias_local:
        melhor = results_df.loc[inst].max()
        ratios.loc[inst] = melhor / results_df.loc[inst]
    
    # Substitui inf/nan por valor alto
    ratios = ratios.replace([np.inf, -np.inf], 1000).fillna(1000)
    
    # Calcula performance profile
    tau_values = np.logspace(0, np.log10(tau_max), 1000)
    
    # Plota
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
             '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
             '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#1a1a1a']
    estilos = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    
    for i, metodo in enumerate(metodos):
        rho = []
        for tau in tau_values:
            prob = np.sum(ratios[metodo] <= tau) / n_problemas
            rho.append(prob)
        
        ax.plot(tau_values, rho, label=metodo, color=cores[i % len(cores)], 
                linestyle=estilos[i % len(estilos)], linewidth=2)
    
    if log_scale:
        ax.set_xscale('log', base=2)
        ax.set_xlabel('τ (escala log₂)', fontsize=12)
    else:
        ax.set_xlabel('τ', fontsize=12)
    
    ax.set_ylabel('P(r ≤ τ)', fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xlim([1, tau_max])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.axvline(x=1, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if arquivo:
        plt.savefig(arquivo, dpi=300, bbox_inches='tight')
        print(f"✓ Salvo: {arquivo}")
    
    plt.show()
    
    # Estatísticas
    stats = pd.DataFrame({
        'Vitórias': [(ratios[m] == 1.0).sum() for m in metodos],
        'ρ(1)': [rho[0] if len(rho) > 0 else 0 for m in metodos]
    }, index=metodos)
    
    return stats

# ============================================================================
# GERAR GRÁFICOS
# ============================================================================

print("="*80)
print("GERANDO PERFORMANCE PROFILES (COM PLI)")
print("="*80)

# 1. Todas as variações + PLI
print("\n1. Todas as variações + PLI (17 métodos)...")
todas = pd.DataFrame({
    'PLI': pli,
    'GRASP-std': grasp_std,
    'GRASP-std+alpha': grasp_std_alpha,
    'GRASP-std+best': grasp_std_best,
    'GRASP-std+hc1': grasp_std_hc1,
    'GRASP-std+hc2': grasp_std_hc2,
    'GRASP-std+hc3': grasp_std_hc3,
    'Tabu-Padrão(FI)': tabu_padrao_fi,
    'Tabu-BestImp': tabu_best_improving,
    'Tabu-Padrão(T2)': tabu_padrao_t2,
    'Tabu-ProbTS': tabu_prob_ts,
    'Tabu-Intensif': tabu_intensif,
    'AG-Padrão': ag_padrao,
    'AG-Padrão+Pop': ag_padrao_pop,
    'AG-Padrão+Mut': ag_padrao_mut,
    'AG-Padrão+Evol1': ag_padrao_evol1,
    'AG-Padrão+Evol2': ag_padrao_evol2,
}, index=instancias)

stats1 = performance_profile(todas, tau_max=5, log_scale=False, 
                             titulo="Performance Profile - Todas Variações + PLI",
                             arquivo="pp_todas_linear.png")

performance_profile(todas, tau_max=10, log_scale=True,
                   titulo="Performance Profile - Todas Variações + PLI (Log)",
                   arquivo="pp_todas_log.png")

print("\nEstatísticas (Todas + PLI):")
print(stats1.sort_values('ρ(1)', ascending=False))

# 2. Melhores + PLI
print("\n2. Melhores variações + PLI (4 métodos)...")
melhores = pd.DataFrame({
    'PLI': pli,
    'GRASP-std+hc2': grasp_std_hc2,
    'Tabu-BestImp': tabu_best_improving,
    'AG-Padrão': ag_padrao,
}, index=instancias)

stats2 = performance_profile(melhores, tau_max=5, log_scale=False,
                             titulo="Performance Profile - Melhores Variações + PLI",
                             arquivo="pp_melhores_linear.png")

performance_profile(melhores, tau_max=10, log_scale=True,
                   titulo="Performance Profile - Melhores Variações + PLI (Log)",
                   arquivo="pp_melhores_log.png")

print("\nEstatísticas (Melhores + PLI):")
print(stats2.sort_values('ρ(1)', ascending=False))

# 3. Apenas Metaheurísticas (SEM PLI) - para comparação direta
print("\n3. Apenas Metaheurísticas - SEM PLI (3 métodos)...")
meta_only = pd.DataFrame({
    'GRASP-std+hc2': grasp_std_hc2,
    'Tabu-BestImp': tabu_best_improving,
    'AG-Padrão': ag_padrao,
}, index=instancias)

stats_meta = performance_profile(meta_only, tau_max=5, log_scale=False,
                                 titulo="Performance Profile - Apenas Metaheurísticas",
                                 arquivo="pp_metaheuristicas_only.png")

print("\nEstatísticas (Apenas Metaheurísticas):")
print(stats_meta.sort_values('ρ(1)', ascending=False))

# 4. GRASP
print("\n4. GRASP - comparação entre variações...")
grasp_df = pd.DataFrame({
    'std': grasp_std,
    'std+alpha': grasp_std_alpha,
    'std+best': grasp_std_best,
    'std+hc1': grasp_std_hc1,
    'std+hc2': grasp_std_hc2,
    'std+hc3': grasp_std_hc3,
}, index=instancias)

performance_profile(grasp_df, tau_max=3, log_scale=False,
                   titulo="Performance Profile - GRASP",
                   arquivo="pp_grasp.png")

# 5. Tabu Search
print("\n5. Tabu Search - comparação entre variações...")
tabu_df = pd.DataFrame({
    'Padrão(FI)': tabu_padrao_fi,
    'BestImp': tabu_best_improving,
    'Padrão(T2)': tabu_padrao_t2,
    'ProbTS': tabu_prob_ts,
    'Intensif': tabu_intensif,
}, index=instancias)

performance_profile(tabu_df, tau_max=3, log_scale=False,
                   titulo="Performance Profile - Tabu Search",
                   arquivo="pp_tabu.png")

# 6. Algoritmo Genético
print("\n6. Algoritmo Genético - comparação entre variações...")
ag_df = pd.DataFrame({
    'Padrão': ag_padrao,
    'Padrão+Pop': ag_padrao_pop,
    'Padrão+Mut': ag_padrao_mut,
    'Padrão+Evol1': ag_padrao_evol1,
    'Padrão+Evol2': ag_padrao_evol2,
}, index=instancias)

performance_profile(ag_df, tau_max=3, log_scale=False,
                   titulo="Performance Profile - Algoritmo Genético",
                   arquivo="pp_ag.png")

print("\n" + "="*80)
print("✓ CONCLUÍDO!")
print("="*80)
print("\nArquivos gerados:")
print("  📊 Com PLI:")
print("    - pp_todas_linear.png (17 métodos)")
print("    - pp_todas_log.png (17 métodos)")
print("    - pp_melhores_linear.png (4 métodos)")
print("    - pp_melhores_log.png (4 métodos)")
print("  📊 Sem PLI:")
print("    - pp_metaheuristicas_only.png (3 métodos)")
print("  📊 Por metaheurística:")
print("    - pp_grasp.png")
print("    - pp_tabu.png")
print("    - pp_ag.png")
print("\n💡 Total: 8 gráficos PNG gerados")
