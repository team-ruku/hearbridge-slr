import os
import random

import numpy as np
import torch
import yaml


def load_config(path="configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if "RecognitionNetwork" in cfg["model"]:
        if "keypoint" in cfg["data"].get("input_streams", ["rgb"]):
            assert (
                "keypoint_s3d" in cfg["model"]["RecognitionNetwork"]
                or "keypoint_resnet3d" in cfg["model"]["RecognitionNetwork"]
            )
            from dataset.Dataset import get_keypoints_num

            keypoints_num = get_keypoints_num(
                keypoint_file=cfg["data"]["keypoint_file"],
                use_keypoints=cfg["data"]["use_keypoints"],
            )
            if "keypoint_s3d" in cfg["model"]["RecognitionNetwork"]:
                cfg["model"]["RecognitionNetwork"]["keypoint_s3d"]["in_channel"] = (
                    keypoints_num
                )
                print(
                    f"Overwrite cfg.model.RecognitionNetwork.keypoint_s3d.in_channel -> {keypoints_num}"
                )
    return cfg


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def neq_load_customized(model, pretrained_dict, verbose=True):
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


def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = move_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list) and type(v[0]) == torch.Tensor:
            batch[k] = [e.to(device) for e in v]
    return batch
