吴恩达Deep Learning 学习笔记



## 1、Logstic Regression(回归)

1、理解w=w-α *df/dw 为什么是这样?

​	由于梯度方向（df/dx，df/dy，df/dw ...）是函数变化最快的方向（通过最大时夹角值得出）

[从为什么梯度方向是函数变化率最快方向详谈梯度下降](https://zhuanlan.zhihu.com/p/420701835)

​	所以每个变量分别取梯度里对应的变化量，整体就沿着下降速度最快的方向去了。

​	然后加上一个学习率来控制步长即可。

2、完整形式

<img src="./assets/image-20230712150336659.png" alt="image-20230712150336659" style="zoom: 67%;" />

<img src="./assets/image-20230712150623806.png" alt="image-20230712150623806" style="zoom:67%;" />

编程Tips : 使用向量化替代for加快速度，利用广播（复制至同规模）