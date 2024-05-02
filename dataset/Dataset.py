import pickle
import json, os, gzip
import numpy as np
from torch.utils import data

Hrnet_Part2index = {
    "pose": list(range(11)),
    "hand": list(range(91, 133)),
    "mouth": list(range(71, 91)),
    "face_others": list(range(23, 71)),
}
for k_ in ["mouth", "face_others", "hand"]:
    Hrnet_Part2index[k_ + "_half"] = Hrnet_Part2index[k_][::2]
    Hrnet_Part2index[k_ + "_1_3"] = Hrnet_Part2index[k_][::3]


def get_keypoints_num(keypoint_file, use_keypoints):
    keypoints_num = 0
    assert "hrnet" in keypoint_file
    Part2index = Hrnet_Part2index
    for k in sorted(use_keypoints):
        keypoints_num += len(Part2index[k])
    return keypoints_num


class ISLRDataset(data.Dataset):
    def __init__(self, dataset_cfg, split):
        super(ISLRDataset, self).__init__()
        self.split = split  # train, dev, test
        self.dataset_cfg = dataset_cfg
        self.root = os.path.join(
            os.path.join("assets", self.dataset_cfg["dataset_name"].split("-")[0])
        )

        if "MSASL" in dataset_cfg["dataset_name"]:
            self.vocab = self.create_vocab()
            self.annotation = self.load_annotations(split)
        elif "WLASL" in dataset_cfg["dataset_name"]:
            self.annotation = self.load_annotations(split)
            self.vocab = self.create_vocab()

        self.input_streams = dataset_cfg.get("input_streams", ["rgb"])
        self.name2keypoints = self.load_keypoints()
        self.word_emb_tab = self.load_word_emb_tab()

    def load_keypoints(self):
        if "keypoint" not in self.input_streams:
            return None

        self.keypoint_file = os.path.join(self.root, self.dataset_cfg["keypoint_file"])
        with open(self.keypoint_file, "rb") as f:
            name2all_keypoints = pickle.load(f)

        print("Keypoints source: hrnet")
        Part2index = Hrnet_Part2index

        name2keypoints = {}
        for name, all_keypoints in name2all_keypoints.items():
            name2keypoints[name] = []
            for k in sorted(self.dataset_cfg["use_keypoints"]):
                selected_index = Part2index[k]
                name2keypoints[name].append(all_keypoints[:, selected_index])  # T, N, 3
            name2keypoints[name] = np.concatenate(
                name2keypoints[name], axis=1
            )  # T, N, 3
            self.keypoints_num = name2keypoints[name].shape[1]

        print(f"Total #={self.keypoints_num}")
        assert self.keypoints_num == get_keypoints_num(
            self.dataset_cfg["keypoint_file"], self.dataset_cfg["use_keypoints"]
        )

        return name2keypoints

    def load_annotations(self, split):
        self.annotation_file = os.path.join(self.root, self.dataset_cfg[split])

        try:
            with open(self.annotation_file, "rb") as f:
                annotation = pickle.load(f)
        except:
            with gzip.open(self.annotation_file, "rb") as f:
                annotation = pickle.load(f)

        if "WLASL" in self.dataset_cfg["dataset_name"]:
            variant_file = os.path.join(
                self.root, self.dataset_cfg["dataset_name"].split("-")[1] + ".json"
            )
            with open(variant_file, "r") as f:
                variant = json.load(f)

            cleaned = []
            for item in annotation:
                if "augmentation" not in item["video_file"] and item["name"] in list(
                    variant.keys()
                ):
                    cleaned.append(item)
            annotation = cleaned
        elif "MSASL" in self.dataset_cfg["dataset_name"]:
            cleaned = []
            for item in annotation:
                if item["label"] in self.vocab:
                    cleaned.append(item)
            annotation = cleaned

        return annotation

    def load_word_emb_tab(self):
        if "word_emb_file" not in self.dataset_cfg:
            return None

        word_emb_file = os.path.join(self.root, self.dataset_cfg["word_emb_file"])
        with open(word_emb_file, "rb") as f:
            word_emb_tab = pickle.load(f)
        return word_emb_tab

    def create_vocab(self):
        if "WLASL" in self.dataset_cfg["dataset_name"]:
            annotation = self.load_annotations("train")
            vocab = []
            for item in annotation:
                if item["label"] not in vocab:
                    vocab.append(item["label"])
            vocab = sorted(vocab)
        elif "MSASL" in self.dataset_cfg["dataset_name"]:
            with open(os.path.join(self.root, "classes.json"), "rb") as f:
                all_vocab = json.load(f)
            num = int(self.dataset_cfg["dataset_name"].split("-")[1])
            vocab = all_vocab[:num]
        return vocab

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        return self.annotation[idx]


def build_dataset(dataset_cfg, split):
    dataset = ISLRDataset(dataset_cfg, split)
    return dataset
