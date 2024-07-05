## gpt2_struct.py
实现gpt2网络过程的代码
不涉及GPU加速
保留了编写过程中运行的测试代码（已注释）

## gpt2_speedup_evaluate.py 
测试不同加速手段的效果
GPTConfig中的is_set_xx控制了不同的加速选项

## gpt2.py
正式的训练代码

## gpt2_loss_evaluate.py
离线评估模型的loss
通过sys.argv[1]加载checkpoint文件
从评估数据集中随机抽取评估数据


