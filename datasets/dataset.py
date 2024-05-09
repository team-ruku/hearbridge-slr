import json
import os
import pickle


class Dataset:
    def __init__(self, config):
        self.config = config
        self.root = os.path.join("assets")

        self.vocab = self.createVocab()
        self.inputStreams = config["input_streams"]
        self.wordEmbTab = self.loadWordEmbTab()

    def loadWordEmbTab(self):
        if "word_emb_file" not in self.config:
            return None

        wordEmbFile = os.path.join(self.root, self.config["word_emb_file"])
        with open(wordEmbFile, "rb") as f:
            wordEmbTab = pickle.load(f)
        return wordEmbTab

    def createVocab(self):
        with open(os.path.join(self.root, "classes.json"), "rb") as f:
            all_vocab = json.load(f)
        num = int(self.config["num"])
        vocab = all_vocab[:num]
        return vocab


def buildDataset(config):
    dataset = Dataset(config)
    return dataset
