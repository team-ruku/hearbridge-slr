import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from dataset.Dataloader import build_dataloader
from dataset.Dataset import build_dataset
from modelling.model import build_model
from utils.metrics import compute_accuracy
from utils.misc import load_config, move_to_device, neq_load_customized, set_seed


def evaluation(
    model,
    dataloader,
    cfg,
    epoch=None,
    global_step=None,
):
    vocab = dataloader.dataset.vocab
    # split = dataloader.dataset.split
    cls_num = len(vocab)

    word_emb_tab = []
    if dataloader.dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataloader.dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg["device"])
    else:
        word_emb_tab = None

    if epoch is not None:
        print(
            "------------------Evaluation epoch={} {} examples #={}---------------------".format(
                epoch, dataloader.dataset.split, len(dataloader.dataset)
            )
        )
    elif global_step is not None:
        print(
            "------------------Evaluation global step={} {} examples #={}------------------".format(
                global_step, dataloader.dataset.split, len(dataloader.dataset)
            )
        )

    model.eval()
    val_stat = defaultdict(float)
    results = defaultdict(dict)
    pred_src = "gloss_logits"
    with torch.no_grad():
        logits_name_lst = []
        for step, batch in enumerate(dataloader):
            batch = move_to_device(batch, cfg["device"])

            forward_output = model(
                is_train=False,
                labels=batch["labels"],
                sgn_videos=batch["sgn_videos"],
                sgn_keypoints=batch["sgn_keypoints"],
                epoch=epoch,
            )
            for k, v in forward_output.items():
                if "_loss" in k:
                    val_stat[k] += v.item()

            # rgb/keypoint/fuse/ensemble_last_logits
            for k, gls_logits in forward_output.items():
                if pred_src not in k or gls_logits is None:
                    continue

                logits_name = k.replace(pred_src, "")
                if "word_fused" in logits_name:
                    continue
                logits_name_lst.append(logits_name)

                decode_output = model.predict_gloss_from_logits(
                    gloss_logits=gls_logits, k=10
                )

                for i in range(decode_output.shape[0]):
                    name = batch["names"][i]
                    hyp = [d.item() for d in decode_output[i]]
                    results[name][f"{logits_name}hyp"] = hyp

                    ref = batch["labels"][i].item()
                    results[name]["ref"] = ref

            print(f"{step + 1} / {len(dataloader)}")

    for k, v in val_stat.items():
        if "_loss" in k:
            print("{} Average:{:.2f}".format(k, v / len(dataloader)))

    per_ins_stat_dict, per_cls_stat_dict = compute_accuracy(
        results, logits_name_lst, cls_num, cfg["device"]
    )

    print(
        "-------------------------Evaluation Finished-------------------------".format()
    )
    return per_ins_stat_dict, per_cls_stat_dict, results


def sync_results(per_ins_stat_dict, per_cls_stat_dict):
    evaluation_results = {}
    for k, per_ins_stat in per_ins_stat_dict.items():
        correct, correct_5, correct_10, num_samples = per_ins_stat
        print("#samples: {}".format(num_samples))
        evaluation_results[f"{k}per_ins_top_1"] = (correct / num_samples).item()
        print(
            "-------------------------{}Per-instance ACC Top-1: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_ins_top_1"]
            )
        )
        evaluation_results[f"{k}per_ins_top_5"] = (correct_5 / num_samples).item()
        print(
            "-------------------------{}Per-instance ACC Top-5: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_ins_top_5"]
            )
        )
        evaluation_results[f"{k}per_ins_top_10"] = (correct_10 / num_samples).item()
        print(
            "-------------------------{}Per-instance ACC Top-10: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_ins_top_10"]
            )
        )

    for k, per_cls_stat in per_cls_stat_dict.items():
        top1_t, top1_f, top5_t, top5_f, top10_t, top10_f = per_cls_stat
        evaluation_results[f"{k}per_cls_top_1"] = np.nanmean(
            (top1_t / (top1_t + top1_f)).cpu().numpy()
        )
        print(
            "-------------------------{}Per-class ACC Top-1: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_cls_top_1"]
            )
        )
        evaluation_results[f"{k}per_cls_top_5"] = np.nanmean(
            (top5_t / (top5_t + top5_f)).cpu().numpy()
        )
        print(
            "-------------------------{}Per-class ACC Top-5: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_cls_top_5"]
            )
        )
        evaluation_results[f"{k}per_cls_top_10"] = np.nanmean(
            (top10_t / (top10_t + top10_f)).cpu().numpy()
        )
        print(
            "-------------------------{}Per-class ACC Top-10: {:.2f}-------------------------".format(
                k, 100 * evaluation_results[f"{k}per_cls_top_10"]
            )
        )

    return evaluation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument("--config", default="configs/default.yaml", type=str)
    parser.add_argument("--ckpt_name", default="best.ckpt", type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["device"] = "cuda:0"
    print(json.dumps(cfg, indent=2))

    torch.cuda.set_device(cfg["device"])
    device = torch.device(cfg["device"])
    set_seed(seed=cfg["training"].get("random_seed", 42))

    dataset = build_dataset(cfg["data"], "train")
    vocab = dataset.vocab
    cls_num = len(vocab)
    word_emb_tab = []
    if dataset.word_emb_tab is not None:
        for w in vocab:
            word_emb_tab.append(torch.from_numpy(dataset.word_emb_tab[w]))
        word_emb_tab = torch.stack(word_emb_tab, dim=0).float().to(cfg["device"])
    else:
        word_emb_tab = None
    del vocab
    del dataset

    model = build_model(cfg, cls_num, word_emb_tab=word_emb_tab)
    load_model_path = os.path.join(
        "assets",
        cfg["data"]["dataset_name"].split("-")[0],
        cfg["data"]["dataset_name"].split("-")[1],
        args.ckpt_name,
    )
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location="cuda")
        neq_load_customized(model, state_dict["model_state"], verbose=True)
        epoch, global_step = (
            state_dict.get("epoch", 0),
            state_dict.get("global_step", 0),
        )
        print("Load model ckpt from " + load_model_path)
    else:
        print(f"{load_model_path} does not exist")
        epoch, global_step = 0, 0

    dataloader, sampler = build_dataloader(cfg, "dev", is_train=False)
    per_ins_stat, per_cls_stat, _ = evaluation(
        model=model,
        dataloader=dataloader,
        cfg=cfg,
        epoch=epoch,
        global_step=global_step,
    )

    sync_results(per_ins_stat, per_cls_stat)
