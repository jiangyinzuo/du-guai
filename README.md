# du-guai

基于机器学习的斗地主算法程序，我们一般只会用两个字来形容这种AI：赌怪！

#### 环境依赖

- python >= 3.7
- numpy
- pytest
- sklearn

#### 启动方式

运行主程序
```bash
cd duguai
python main.py
```

运行脚本文件
```bash
cd script

# Q-Learning模型训练
python q_learning.py
```

#### 项目目录说明

- duguai：程序源代码目录。main.py是入口程序
- dataset：数据集目录
- doc：项目文档
- notebook：ipython notebook目录
- script：一些脚本程序（如数据生成脚本）
- test：单元测试

#### 配置说明
配置文件为duguai/.env

调试/生产模式：mode = debug/prod
开启/关闭测试：test = on/off

#### 斗地主规则
[RULES.md](./RULES.md)

#### 训练结果
训练时长：788秒
训练次数：10100次
