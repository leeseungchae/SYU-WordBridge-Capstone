import json


def json_loads(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def get_target_modules(model):
    target_modules = []
    for name, module in model.named_modules():
        if any(substr in name for substr in ["self_attn", "encoder_attn"]):
            if any(layer in name for layer in ["k_proj", "v_proj", "q_proj", "out_proj"]):
                target_modules.append(name)
    return target_modules
