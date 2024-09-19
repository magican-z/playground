import sys
import os
import json
import argparse
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import LabeledFewShot
from dspy.teleprompt import BootstrapFewShot
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt import BootstrapFewShotWithOptuna



class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


def use_local_cpm():
    # 这里连接了一个本地部署的MiniCPM3, 可以根据需要修改成自己的模型，远程连接其他模型参考GML_Client.py
    api_base = os.environ.get("MINICPM3_API_BASE", "EMPTY")
    print(f'using minicpm3 api {api_base}')
    gpt_interface = dspy.OpenAI(model='gpt-3.5-turbo-1106',
                                api_base=api_base,
                                api_key='empty',
                                max_tokens=300)
    dspy.configure(lm=gpt_interface)
    return gpt_interface


def run_LabeledFewShot(n_examples):
    lm = use_local_cpm()
    
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:50]

    teleprompter = LabeledFewShot(k=n_examples)
    optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    evaluate(optimized_cot)
    #lm.inspect_history(n=15)


def run_BootstrapFewShot(n_examples):
    lm = use_local_cpm()

    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:50]

    config = dict(max_labeled_demos=n_examples, max_bootstrapped_demos=n_examples)
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    raw_cot = CoT()
    optimized_cot = teleprompter.compile(raw_cot, trainset=gsm8k_trainset)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    evaluate(optimized_cot)
    #lm.inspect_history(n=15)


def run_BootstrapFewShotWithRandomSearch(n_examples):
    lm = use_local_cpm()

    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:50], gsm8k.dev[:50]

    config = dict(max_labeled_demos=n_examples, max_bootstrapped_demos=n_examples)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
    raw_cot = CoT()
    optimized_cot = teleprompter.compile(raw_cot, trainset=gsm8k_trainset)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    evaluate(optimized_cot)


def run_BootstrapFewShotWithOptuna(n_examples):
    lm = use_local_cpm()

    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:50], gsm8k.dev[:50]

    config = dict(max_labeled_demos=n_examples, max_bootstrapped_demos=n_examples)
    teleprompter = BootstrapFewShotWithOptuna(metric=gsm8k_metric, **config)
    raw_cot = CoT()
    optimized_cot = teleprompter.compile(raw_cot, trainset=gsm8k_trainset, max_demos=n_examples)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    evaluate(optimized_cot)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FewShot优化演示')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='Optimization Mode ')
    args = parser.parse_args()
    
    if args.mode == 'LabeledFewShot':
        run_LabeledFewShot(3)
    elif args.mode == 'BootstrapFewShot':
        run_BootstrapFewShot(3)
    elif args.mode == 'BootstrapFewShotWithRandomSearch':
        run_BootstrapFewShotWithRandomSearch(3)
    elif args.mode == 'BootstrapFewShotWithOptuna':
        run_BootstrapFewShotWithOptuna(3)
    else:
        print('Invalid mode')
