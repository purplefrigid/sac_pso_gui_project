import os
import re
import matplotlib.pyplot as plt
from pso import PSOOptimizer
from Optimization_Algorithm  import GAOptimizer,SAOptimizer,ACOOptimizer

def extract_scores_from_file(file_path):
    steps = []
    scores = []
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # 使用正则表达式匹配所有step和对应的global_best_score
        step_matches = re.finditer(r'Step:\s*(\d+).*?self\.global_best_score:([-\d.]+)', content, re.DOTALL)
        
        for match in step_matches:
            step = int(match.group(1))
            score = float(match.group(2))
            steps.append(step)
            scores.append(score)
    
    return steps, scores

def plot_all_files(folder_path):
    plt.figure(figsize=(12, 8))
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # 遍历文件夹中的所有txt文件
    for filename in files:
        if filename.endswith('.txt'):
            name, _ = os.path.splitext(filename)
            file_path = os.path.join(folder_path, filename)
            steps, scores = extract_scores_from_file(file_path)
            
            if steps and scores:  # 确保有数据
                # 绘制折线图，使用文件名作为图例
                plt.plot(steps, scores, marker='o', label=name)
    # optimizer = PSOOptimizer("库.xlsx", num_particles=10, max_iter=10)  
    # optimizer.optimize() 
    optimizer1 = PSOOptimizer("库.xlsx", num_particles=20, max_iter=10)  
    optimizer1.optimize() 
    ga_optimizer = GAOptimizer("库.xlsx", population_size=20, max_iter=10)
    sa_optimizer = SAOptimizer("库.xlsx", max_iter=10)
    aco_optimizer = ACOOptimizer("库.xlsx", num_ants=20, max_iter=11)

    
    # 执行优化
    ga_best_params, ga_best_score = ga_optimizer.optimize()
    sa_best_params, sa_best_score = sa_optimizer.optimize()
    aco_best_params, aco_best_score = aco_optimizer.optimize()

    plt.xlabel('Step')
    plt.ylabel('Global Best Score')
    plt.title('Global Best Score Evolution Across Files')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在图表外部右侧
    # plt.grid(True)
    plt.tight_layout()  # 调整布局防止标签重叠
    plt.show()

# 使用示例
folder_path = 'C:\\Users\\pc\\Desktop\\新建文件夹\\test4'  # 替换为你的文件夹路径
plot_all_files(folder_path)