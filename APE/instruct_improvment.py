import sys
import os
import json
import argparse
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
from dspy.teleprompt import COPRO
from dspy.teleprompt import MIPROv2
from GLM_Client import GLM


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


def run_COPRO_self():
    lm = use_local_cpm()
    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:50]

    teleprompter = COPRO(metric=gsm8k_metric)
    kwargs = dict(num_threads=4, display_progress=True, display_table=0)
    raw_cot = CoT()
    compiled_prompt_opt = teleprompter.compile(raw_cot.deepcopy(), trainset=gsm8k_trainset, eval_kwargs=kwargs)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    eval_score = evaluate(compiled_prompt_opt, devset=gsm8k_devset, **kwargs)
    print(eval_score)
    #lm.inspect_history(n=100)


def run_COPRO_GML():
    lm = use_local_cpm()
    api_key = os.environ.get("GLM_API_KEY", "")
    print(f'using glm key {api_key}')
    glm = GLM(model='glm-4', api_key=api_key)

    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:50]

    teleprompter = COPRO(prompt_model=glm, metric=gsm8k_metric)
    kwargs = dict(num_threads=4, display_progress=True, display_table=0)
    raw_cot = CoT()
    compiled_prompt_opt = teleprompter.compile(raw_cot.deepcopy(), trainset=gsm8k_trainset, eval_kwargs=kwargs)

    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    eval_score = evaluate(compiled_prompt_opt, devset=gsm8k_devset, **kwargs)
    print(eval_score)


def run_MIPROv2():
    lm = use_local_cpm()

    gsm8k = GSM8K()
    gsm8k_trainset, gsm8k_devset = gsm8k.train[:120], gsm8k.dev[:50]

    teleprompter = MIPROv2(prompt_model=lm, task_model=lm, metric=gsm8k_metric, num_candidates=10, init_temperature=1.0)
    kwargs = dict(num_threads=4, display_progress=True, display_table=0)
    raw_cot = CoT()
    compiled_prompt_opt = teleprompter.compile(raw_cot, trainset=gsm8k_trainset, num_batches=30, max_bootstrapped_demos=3, max_labeled_demos=3, eval_kwargs=kwargs)
    
    evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    eval_score = evaluate(compiled_prompt_opt, devset=gsm8k_devset, **kwargs)
    print(eval_score)



if __name__ == '__main__':
    #run_COPRO_self()
    #run_COPRO_GML()
    run_MIPROv2()
