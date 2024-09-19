import os
from dsp import LM
from zhipuai import ZhipuAI


class GLM(LM):
    def __init__(self, model, api_key, **kwargs):
        self.model = model
        self.api_key = api_key
        self.provider = "GLM"
        self.client = ZhipuAI(api_key=api_key)
        self.history = []
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }

    def basic_request(self, prompt: str, **kwargs):
        n = kwargs.pop('n', 1)
        resp = []
        for i in range(n):
            response = self.client.chat.completions.create(
                    model=self.model,  
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs)
            item = self.change_to_inspect(response)
            resp.append(item)
        self.history.append({
            "prompt": prompt,
            "response": {"choices": resp},
            "kwargs": kwargs,
        })
        return resp
    
    def change_to_inspect(self, response):
        choice = response.choices[0]
        return {
            "role": choice.message.role,
            "text": choice.message.content,
        }

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)
        completions = [resp["text"] for resp in response]

        return completions
    

if __name__ == "__main__":
    api_key = os.environ.get("GLM_API_KEY", "")
    print(f'using glm key {api_key}')
    model = GLM(model='glm-4-flash', api_key=api_key)
    print(model('你好，你是谁？'))
