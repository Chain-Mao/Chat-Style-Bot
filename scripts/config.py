import json
import sys

def load_config(model_type):
    with open('config/wechat/settings.json', 'r') as f:
        config: dict = json.load(f)
        if model_type == 'llama3':
            config = {**config['llama3_args']}
        elif model_type == 'qwen2':
            config = {**config['qwen2_args']}
        elif model_type == 'glm4':
            config = {**config['glm4_args']}
        else:
            raise ValueError('暂不支持的该模型，但您可以自行仿照已有代码扩展该模型')

    sys.argv += dict_to_argv(config)

    return config

def dict_to_argv(d):
    argv = []
    for k, v in d.items():
        argv.append('--' + k)
        if v is not None:
            argv.append(str(v))
    return argv