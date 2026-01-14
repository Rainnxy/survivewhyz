import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.collections import LineCollection

# 1. 定义洛伦兹方程
def lorenz(state, t):
    x, y, z = state
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# 2. 设置初始状态和时间点
state0 = [0.1, 0.0, 0.0]
t = np.arange(0.0, 300.0, 0.01) # 时间越长，轨迹越丰富

# 3. 求解微分方程
states = odeint(lorenz, state0, t)
x = states[:, 0]
z = states[:, 2] # 取 x-z 平面投影，这是最经典的蝴蝶形状

# 4. 绘图设置
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off') # 关闭坐标轴

# 5. 创建渐变色轨迹
# 将点转换为线段集合
points = np.array([x, z]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# 设置颜色映射 (winter 是从蓝到青的渐变，非常符合 elegantbook 的色调)
norm = plt.Normalize(t.min(), t.max())
lc = LineCollection(segments, cmap='winter', norm=norm, alpha=0.8, linewidth=1.0)
lc.set_array(t)

ax.add_collection(lc)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(z.min(), z.max())

# 6. 保存图片 (透明背景)
plt.savefig('lorenz_attractor.png', transparent=True, bbox_inches='tight', dpi=300)
print("图片已生成：lorenz_attractor.png")