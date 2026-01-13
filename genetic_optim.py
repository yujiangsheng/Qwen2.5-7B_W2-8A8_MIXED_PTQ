"""
遗传算法优化器 (Genetic Algorithm Optimizer)
==========================================

本模块实现用于混合精度量化配置优化的遗传算法（GA）。

核心思想：
---------
将每层的量化位数（W2/W4/W8）视为基因，通过进化选择找到最优的混合精度配置：
- 目标1：最大化模型压缩率
- 目标2：最小化量化精度损失

遗传算法流程：
-------------
1. 初始化种群：随机生成多个位宽配置
2. 适应度评估：基于层敏感度计算总MSE
3. 选择：保留适应度高的个体
4. 交叉：两个父代生成子代
5. 变异：随机改变部分基因
6. 迭代：重复2-5直到收敛

参数说明：
---------
- population_size: 种群大小，越大搜索越全面但速度越慢
- n_generations: 迭代代数，越多结果越好但耗时更长
- mutation_rate: 变异率，控制探索力度
- target_compression: 目标压缩比，约束条件
"""

import numpy as np
from typing import Callable, Dict, List, Tuple


class MixedPrecisionGA:
    """
    混合精度遗传算法优化器
    
    用于找到最优的逐层量化位宽配置，平衡模型大小和精度。
    
    属性：
    -----
    n_layers : int
        需要优化的层数
    pop_size : int
        种群大小
    n_generations : int
        迭代代数
    mutation_rate : float
        变异率 (0.0~1.0)
    bit_options : list
        可选的位宽列表 [2, 4, 8]
    
    示例：
    ------
    >>> ga = MixedPrecisionGA(n_layers=196, population_size=20)
    >>> best_config = ga.optimize(fitness_func, target_compression=0.3)
    >>> print(f"最优配置: {best_config}")
    """
    
    def __init__(self, n_layers: int, population_size: int = 20, 
                 n_generations: int = 15, mutation_rate: float = 0.15):
        """
        初始化遗传算法优化器
        
        参数：
        -----
        n_layers : int
            待优化的层数（Qwen2.5-7B共196个线性层）
        population_size : int
            种群大小，建议15-30
        n_generations : int
            进化代数，建议10-20
        mutation_rate : float
            变异率，建议0.1-0.2
        """
        self.n_layers = n_layers
        self.pop_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        
        # 可选位宽: W2(激进压缩), W4(平衡), W8(近似FP16)
        self.bit_options = [2, 4, 8]
    
    def initialize_population(self) -> List[np.ndarray]:
        """
        初始化种群
        
        使用加权随机初始化，偏向W4以保证初始质量：
        - W2: 30% 概率（激进压缩层）
        - W4: 50% 概率（平衡层）
        - W8: 20% 概率（敏感层）
        
        返回：
        ------
        List[np.ndarray]
            种群列表，每个个体是一个位宽配置数组
        """
        population = []
        for _ in range(self.pop_size):
            # 加权随机初始化
            individual = np.random.choice(
                self.bit_options, 
                size=self.n_layers, 
                p=[0.3, 0.5, 0.2]  # W2, W4, W8的概率
            )
            population.append(individual)
        return population
    
    def compute_model_size(self, individual: np.ndarray) -> float:
        """
        计算相对模型大小
        
        参数：
        -----
        individual : np.ndarray
            位宽配置数组
        
        返回：
        ------
        float
            相对于FP16的大小比例 (0.0~1.0)
        """
        return np.sum(individual) / (self.n_layers * 16)  # 相对于FP16
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        单点交叉操作
        
        随机选择一个交叉点，将两个父代的基因组合：
        child = [parent1[:point], parent2[point:]]
        
        参数：
        -----
        parent1, parent2 : np.ndarray
            两个父代个体
        
        返回：
        ------
        np.ndarray
            子代个体
        """
        point = np.random.randint(1, self.n_layers)
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        变异操作
        
        以mutation_rate的概率随机改变每个基因
        
        参数：
        -----
        individual : np.ndarray
            待变异个体
        
        返回：
        ------
        np.ndarray
            变异后的个体
        """
        for i in range(self.n_layers):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.choice(self.bit_options)
        return individual
    
    def optimize(self, fitness_func: Callable[[np.ndarray], float], 
                 target_compression: float = 0.3) -> np.ndarray:
        """
        执行遗传算法优化
        
        参数：
        -----
        fitness_func : Callable
            适应度函数，输入位宽配置，返回质量分数（越高越好）
            通常使用负MSE作为适应度
        target_compression : float
            目标压缩比（相对于FP16），默认0.3表示30%大小
        
        返回：
        ------
        np.ndarray
            最优位宽配置
        
        示例：
        ------
        >>> def fitness(config):
        ...     return -compute_mse(model, config)  # 负MSE作为适应度
        >>> best = ga.optimize(fitness, target_compression=0.25)
        """
        population = self.initialize_population()
        best_individual = None
        best_score = -float('inf')
        
        for gen in range(self.n_generations):
            scores = []
            
            for indiv in population:
                # 计算质量分数
                quality = fitness_func(indiv)
                
                # 大小惩罚：如果超过目标压缩比，施加惩罚
                size_ratio = self.compute_model_size(indiv)
                size_penalty = max(0, (size_ratio - target_compression) * 10)
                
                # 综合分数 = 质量 - 大小惩罚
                score = quality - size_penalty
                scores.append(score)
                
                # 更新最优解
                if score > best_score:
                    best_score = score
                    best_individual = indiv.copy()
            
            # === 选择操作 ===
            # 保留适应度最高的50%个体
            sorted_idx = np.argsort(scores)[::-1]
            survivors = [population[i] for i in sorted_idx[:self.pop_size // 2]]
            
            # === 生成新种群 ===
            new_pop = survivors[:]
            while len(new_pop) < self.pop_size:
                # 随机选择两个父代
                p1, p2 = np.random.choice(len(survivors), 2, replace=False)
                # 交叉
                child = self.crossover(survivors[p1], survivors[p2])
                # 变异
                child = self.mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
            # 打印进度
            self._print_generation_stats(gen, best_individual, best_score)
        
        return best_individual
    
    def _print_generation_stats(self, gen: int, best: np.ndarray, score: float):
        """打印每代的统计信息"""
        w2_count = np.sum(best == 2)
        w4_count = np.sum(best == 4)
        w8_count = np.sum(best == 8)
        size = self.compute_model_size(best)
        print(f"第{gen+1}代: 分数={score:.4f}, 大小={size:.1%}, "
              f"W2={w2_count}, W4={w4_count}, W8={w8_count}")


class LayerSensitivityAnalyzer:
    """
    层敏感度分析器
    
    评估每层对不同量化位宽的敏感程度，用于指导混合精度配置。
    
    敏感度定义：
    -----------
    敏感度 = MSE(量化输出, 原始输出)
    
    敏感度越高的层应使用更高的位宽，敏感度低的层可使用W2激进压缩。
    
    敏感度分类：
    -----------
    - 低敏感度 (MSE < 0.1): 可安全使用W2
    - 中敏感度 (0.1 <= MSE < 0.5): 建议使用W4
    - 高敏感度 (MSE >= 0.5): 应保持W8
    """
    
    def __init__(self, bit_options: List[int] = None):
        """
        初始化敏感度分析器
        
        参数：
        -----
        bit_options : List[int]
            要测试的位宽列表，默认 [2, 4, 8]
        """
        self.bit_options = bit_options or [2, 4, 8]
    
    def analyze(self, layer, calib_input, quantize_fn: Callable) -> Dict[int, float]:
        """
        分析单层对不同位宽的敏感度
        
        参数：
        -----
        layer : nn.Module
            待分析的线性层
        calib_input : torch.Tensor
            校准输入
        quantize_fn : Callable
            量化函数
        
        返回：
        ------
        Dict[int, float]
            位宽到MSE的映射，如 {2: 0.15, 4: 0.05, 8: 0.01}
        """
        import torch
        
        # 获取原始输出
        original_output = layer(calib_input)
        
        sensitivity = {}
        for n_bits in self.bit_options:
            # 量化权重
            w = layer.weight
            limit = w.abs().amax() * 0.9  # 裁剪到90%
            w_clipped = torch.clamp(w, -limit, limit)
            w_q = quantize_fn(w_clipped, n_bits=n_bits, group_size=128, sym=True)
            
            # 量化激活（A8固定）
            x_q = quantize_fn(calib_input, n_bits=8, group_size=-1, sym=False)
            
            # 计算量化后输出
            out_q = torch.nn.functional.linear(x_q, w_q, layer.bias)
            
            # 计算MSE
            mse = torch.mean((out_q - original_output) ** 2).item()
            sensitivity[n_bits] = mse
        
        return sensitivity
    
    def classify_sensitivity(self, mse_w2: float) -> str:
        """
        根据W2的MSE对敏感度进行分类
        
        参数：
        -----
        mse_w2 : float
            W2量化的MSE
        
        返回：
        ------
        str
            敏感度类别: "低敏感度(W2)", "中敏感度(W4)", "高敏感度(W8)"
        """
        if mse_w2 < 0.1:
            return "低敏感度(可用W2)"
        elif mse_w2 < 0.5:
            return "中敏感度(建议W4)"
        else:
            return "高敏感度(保持W8)"
