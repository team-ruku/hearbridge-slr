import torch
import numpy as np


def compute_accuracy(
    results,
    logits_name_lst,
    cls_num,
    device,
    name_lst=[],
    effective_label_idx=[],
):
    per_ins_stat_dict, per_cls_stat_dict = {}, {}
    for k in logits_name_lst:
        correct = correct_5 = correct_10 = num_samples = 0
        top1_t = np.zeros(cls_num, dtype=np.int32)
        top1_f = np.zeros(cls_num, dtype=np.int32)
        top5_t = np.zeros(cls_num, dtype=np.int32)
        top5_f = np.zeros(cls_num, dtype=np.int32)
        top10_t = np.zeros(cls_num, dtype=np.int32)
        top10_f = np.zeros(cls_num, dtype=np.int32)
        for name in results.keys():
            if len(name_lst) > 0 and name not in name_lst:
                continue

            res = results[name]
            # if len(cfg['data']['input_streams']) == 1:
            #     hyp_lst = res['hyp']
            # elif len(cfg['data']['input_streams']) > 1:
            #     hyp_lst = res['ensemble_last_hyp']
            hyp = res[f"{k}hyp"]
            # update hyp list
            if len(effective_label_idx) > 0:
                hyp_lst = []
                for h in hyp:
                    if h in effective_label_idx:
                        hyp_lst.append(h)
                        effective_label_idx.remove(h)
            else:
                hyp_lst = hyp
            ref = res["ref"]

            if ref == hyp_lst[0]:
                correct += 1
                top1_t[ref] += 1
            else:
                top1_f[ref] += 1

            if ref in hyp_lst[:5]:
                correct_5 += 1
                top5_t[ref] += 1
            else:
                top5_f[ref] += 1

            if ref in hyp_lst[:10]:
                correct_10 += 1
                top10_t[ref] += 1
            else:
                top10_f[ref] += 1
            num_samples += 1

        per_ins_stat = torch.tensor(
            [correct, correct_5, correct_10, num_samples],
            dtype=torch.float32,
            device=device,
        )
        per_cls_stat = (
            torch.stack(
                [
                    torch.from_numpy(top1_t),
                    torch.from_numpy(top1_f),
                    torch.from_numpy(top5_t),
                    torch.from_numpy(top5_f),
                    torch.from_numpy(top10_t),
                    torch.from_numpy(top10_f),
                ],
                dim=0,
            )
            .float()
            .to(device)
        )
        per_ins_stat_dict[k] = per_ins_stat
        per_cls_stat_dict[k] = per_cls_stat
    return per_ins_stat_dict, per_cls_stat_dict
