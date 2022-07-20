from flair.embeddings import WordEmbeddings, FlairEmbeddings, TransformerWordEmbeddings, StackedEmbeddings

from itertools import combinations as combinations
from itertools import product as product
import numpy as np
import json
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import torch 
import random

def DownloadFewRel():
    url = "https://raw.githubusercontent.com/ylaxor/fewrel-data/main/train.json"
    df = pd.read_json(url)
    corpus = []
    for column_name, column_values in df.iteritems():
        column_values = column_values.tolist()
        for instance in column_values:
                sentence = " ".join(instance["tokens"])
                head_span = instance["h"][2][0]
                tail_span = instance["t"][2][0]
                direction = "left-to-right" if head_span[0] < tail_span[0] else "right-to-left"
                sample = {"sentence": sentence, "head":{"start":head_span[0],"end":head_span[-1]+1}, "tail":{"start":tail_span[0],"end":tail_span[-1]+1}, "class":column_name, "direction":direction}
                corpus.append(sample)
    return corpus

def ReadFewRelFromFile(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
        corpus = []
        for relation_type, instances in data.items():
            for instance in instances:
                sentence = " ".join(instance["tokens"])
                head_span = instance["h"][2][0]
                tail_span = instance["t"][2][0]
                direction = "left-to-right" if head_span[0] < tail_span[0] else "right-to-left"
                sample = {"sentence": sentence, "head":{"start":head_span[0],"end":head_span[-1]+1}, "tail":{"start":tail_span[0],"end":tail_span[-1]+1}, "class":relation_type, "direction":direction}
                corpus.append(sample)
        return corpus
    except:
        raise Exception("Could not read file, or file is not a valid json file")
     
def LoadFewRelData(use_local_file=False, path=None, usage="demo-with-three-relations"):
    """Loads samples from Google FewRel semantic relation extraction dataset.

    Parameters:
        :use_local_file: If True, loads samples from local file specified in path. Default is False, which downloads samples from Github.
        :path: Path to local file containing original FewRel data. Required if use_local_file is True.
        :usage: 'demo-with-three-relations' mode loads the subset of FewRel samples that belong to the following three pre-chosen relations: 'Has nationality', 'Located in', and 'Plays'. Set this param to 'all-available-relations' to work with all the available (+100) relations.
    
    Returns:
        :dict: A dict object with three fields: train, dev and test. Each field contains a list of the requested pre-annotated samples.
    """
    assert usage in ["demo-with-three-relations", "all-available-relations"]
    if isinstance(path, str) and use_local_file:
        relation_classif_corpus_full = ReadFewRelFromFile(path)
    else:
        relation_classif_corpus_full = DownloadFewRel()
    demo_classes = ['P131', 'P1303', 'P27']
    names_corrections = {'P131': 'LOCATED-IN', 'P1303': 'PLAYS', 'P27': 'HAS-NATIONALITY'}
    relation_classif_corpus_demo = [s.copy() for s in relation_classif_corpus_full if s["class"] in demo_classes]
    for s in relation_classif_corpus_demo:
        s["class"] = names_corrections[s["class"]]
    if usage == "demo-with-three-relations":
        corpus = relation_classif_corpus_demo
    if usage == "all-available-relations":
        corpus = relation_classif_corpus_full
    grouped_by_class = pd.DataFrame(corpus).groupby(["class"])
    train_data = []
    dev_data = []
    test_data = []
    for g in grouped_by_class:
        group_samples = g[1].values.tolist()
        group_samples = list(map(lambda s: {"sentence":s[0], "head":s[1], "tail":s[2], "class":s[3], "direction":s[4]}, group_samples))
        g_learn, g_test = train_test_split(group_samples, test_size=0.2, random_state=0)
        g_train, g_dev = train_test_split(g_learn, test_size=0.2, random_state=0)
        train_data.extend(g_train)
        dev_data.extend(g_dev)
        test_data.extend(g_test)
    return {"train":train_data, "dev":dev_data, "test":test_data}

def SeedEverything(seed=1):
    """Seeds random, numpy.random and torch modules with the given seed for reproducibility of experiments.

    Parameters:
        :seed: int.

    Returns:
        nothing.
    """
    assert isinstance(seed, int)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def GetAvailableDevice():
    available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return available_device

def LoadCustomCorpus(file_path=None):
    """Loads a custom data set of pre-annotated samples from a valid jsonl file.
    A valid jsonl file is a text file that ends with .jsonl. And each line is a json object like the following:
    {"id":5,"text":"In 2014 , Emma represented Italy with the song \" La mia città \" , finishing in 21st place .","entities":[{"id":7,"label":"ENTITY","start_offset":10,"end_offset":14},{"id":8,"label":"ENTITY","start_offset":27,"end_offset":32}],"relations":[{"id":4,"from_id":7,"to_id":8,"type":"HAS NATIONALITY"}]}


    Parameters:
        :file_path: path to a valid jsonl file.
    
    Returns:
        A list of dict objects. Each dict object represents a loaded sample and contains the following fields: sentence, head, tail, class, and direction.
    """
    import json
    try:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]
    except:
        raise Exception("Could not load the jsonl file.")
    samples = []
    for sample in data:
        entities = sample["entities"]
        text = sample["text"]
        subsamples = []
        for r in sample["relations"]:
            relation_type = r["type"]
            head_id = min(r["from_id"], r["to_id"])
            tail_id = max(r["from_id"], r["to_id"])
            relation_direction = "left-to-right" if head_id == r["from_id"] else "right-to-left"
            head_entity = [e for e in entities if e["id"] == head_id][0]
            tail_entity = [e for e in entities if e["id"] == tail_id][0]
            before_head_chunk = text[:head_entity["start_offset"]]
            head_chunk = text[head_entity["start_offset"]:head_entity["end_offset"]]
            after_head_chunk = text[head_entity["end_offset"]:tail_entity["start_offset"]]
            tail_chunk = text[tail_entity["start_offset"]:tail_entity["end_offset"]]
            after_tail_chunk = text[tail_entity["end_offset"]:]
            Head = {"start": len(before_head_chunk.split()), "end": len(before_head_chunk.split()) + len(head_chunk.split())}
            Tail = {"start": len(before_head_chunk.split()) + len(head_chunk.split()) + len(after_head_chunk.split()), "end": len(before_head_chunk.split()) + len(head_chunk.split()) + len(after_head_chunk.split()) + len(tail_chunk.split())}
            formatted_sample = {"sentence": text, "head": Head, "tail": Tail, "class": relation_type, "direction": relation_direction}
            subsamples.append(formatted_sample)
        samples.extend(subsamples)
    return samples

def LoadSignleEmbedder(name, props):
        family = props["family"]
        args = props["args"] if "args" in props else {}
        if family == "Word":
            embedder  = WordEmbeddings
        elif family == "Flair":
            embedder = FlairEmbeddings
        elif family == "Transformer": 
            embedder = TransformerWordEmbeddings
        else:
            raise ValueError("Unknown embedding family: {}".format(family))
        try:
            return embedder(name, **args)
        except Exception as e:
            raise ValueError("Failed to load embedding {} from {} embedders family: {}".format(name, family, e))

def EmbeddingsLoader(requested_embeddings=None):
    """Fetches the embeddings specified in the requested_embeddings parameter into some flair.embeddings.StackedEmbeddings object.

        Parameters:
            :requested_embeddings: A dict containing the desired embeddings' configurations. Required fields are:

                correct name of the embedding model as a key fot the dictionary.
                
                "family": str. The family of the embedding model. Can be "Word", "Flair" or "Transformer".
                
                "args": dict. The embedding model's configuration. See the documentation of the desired embedding in Flair or HFT for more details. Those are easy to find and can be directly applied here.
                
                For example:
                    - {"glove": {"family": "Word", "args": {}}}
                    - {"bert-base-uncased": {"family": "Transformer", "args": {"allow_long_sentences": True}}}

        Returns:
            :embeddings: A stack of the desired embedding models as torch modules wrapped into a flair.embeddings.StackedEmbeddings object.
    """

    try:
            assert isinstance(requested_embeddings, dict) and len(requested_embeddings) > 0
            assert all(isinstance(name, str) for name in requested_embeddings.keys())
            assert all(props["family"] in ["Word", "Flair", "Transformer"] for props in requested_embeddings.values())
            assert all(isinstance(props["args"], dict) for props in requested_embeddings.values() if "args" in props)
            embeddings = StackedEmbeddings([LoadSignleEmbedder(name, props) for name, props in requested_embeddings.items()])
            return embeddings
    except AssertionError:
            raise ValueError("Invalid embeddings configuration: {}".format(requested_embeddings))
    
