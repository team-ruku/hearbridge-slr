import os
import random

import numpy as np
import torch
import yaml


def loadConfig(path="configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if "RecognitionNetwork" in config["model"]:
        if "keypoint" in config["data"].get("input_streams", ["rgb"]):
            assert (
                "keypoint_s3d" in config["model"]["RecognitionNetwork"]
                or "keypoint_resnet3d" in config["model"]["RecognitionNetwork"]
            )

            if "keypoint_s3d" in config["model"]["RecognitionNetwork"]:
                config["model"]["RecognitionNetwork"]["keypoint_s3d"]["in_channel"] = 63

    return config


def loadCustomized(model, pretrained_dict, verbose=True):
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        # print(list(model_dict.keys()))
        print("\n=======Check Weights Loading======")
        print("Weights not used from pretrained file:")
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            tmp[k] = v
        else:
            if verbose:
                print(k)
    if verbose:
        print("---------------------------")
        print("Weights not loaded into new model:")
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
            elif model_dict[k].shape != pretrained_dict[k].shape:
                print(k, "shape mis-matched, not loaded")
        print("===================================\n")

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model


def moveToDevice(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = moveToDevice(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list) and type(v[0]) == torch.Tensor:
            batch[k] = [e.to(device) for e in v]
    return batch


def setSeed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
