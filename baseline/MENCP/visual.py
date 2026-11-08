import numpy as np  
from sklearn.manifold import TSNE  
import matplotlib.pyplot as plt  
from matplotlib.animation import FuncAnimation  
  
# 假设你有一个按时间排序的向量列表 vectors_over_time，其中每个元素是一个NumPy数组  
# 例如: vectors_over_time = [vector1, vector2, ..., vectorN]，其中每个vector都是一个形状为(num_features,)的数组  
  
# 这里我们使用随机数据作为示例  
num_time_steps = 10  
num_features = 50  
num_samples = 100  
vectors_over_time = [np.random.rand(num_samples, num_features) for _ in range(num_time_steps)]  
  
# 初始化一个用于保存降维结果的列表  
tsne_results = []  
  
# 应用t-SNE到每个时间步的向量集上  
for t, vectors in enumerate(vectors_over_time):  
    tsne = TSNE(n_components=2, random_state=t)  # 使用不同的随机状态以确保每次结果都不同  
    tsne_result = tsne.fit_transform(vectors)  
    tsne_results.append(tsne_result)  
  
# 创建一个用于动画绘图的函数  
def animate(i):  
    ax.clear()  
    scatter = ax.scatter(tsne_results[i][:, 0], tsne_results[i][:, 1])  
    title = ax.set_title(f'Time Step {i}')  
    return scatter, title  
  
# 创建一个图形和轴对象  
fig, ax = plt.subplots()  
  
# 创建一个动画对象  
ani = FuncAnimation(fig, animate, frames=len(tsne_results), interval=200, blit=True)  
  
# 显示动画  
plt.show()