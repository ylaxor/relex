from pathlib import Path
from flair.data import Corpus
from flair.datasets import ColumnCorpus
import logging
import sys
from .relex_dataset import RelexDataset

class RelexCorpus:
    """RelexCorpus class to wrap learning pre-annotated sentences with respect to some semantic relations. Used during learning loops of both the relex.models.EntitiesTagger and relex.models.PairClassifier modules whose roles are respectively to tag entities and predict the relation (it's type and/or direction) between pairs of them. It puts given samples in relex.data.RelexDataset instances that would be loaded during the learning process of the relation classifier (relex.models.PairClassifier) module. For that, it alse computes some useful labelling mappings for present relations types and directions. Additionally, it provides useful methods to generate a tagging flair.data.Corpus object along with useful tags mappings useful for the tagging task ensured by the relex.models.EntitiesTagger module.
    Samples in each param list must be dictionaries with the following items:
                - 'sentence': a string containing the sentence.
                - 'class': a string containing the class of the relation.
                - 'direction': a string containing the direction of the relation.
                - 'head': a dictionary containing {'start': int, 'end': int} information of head entity.
                - 'tail': a dictionary containing {'start': int, 'end': int} information of tail entity.
        
    
    Parameters:
        :train (list): training set of pre-annotated sentences.
        :dev (list): validation set of pre-annotated sentences.
        :test (list): testing set of pre-annotated sentences.
    
    Attributes:
        :train (): :class: `relex.data.RelexDataset` training torch-wrapped data set. Used in relation classification task.
        :dev (relex.data.RelexDataset): validation torch-wrapped data set. Used in relation classification task.
        :test (relex.data.RelexDataset): testing torch-wrapped data set. Used in relation classification task.
        :class2idx (dict): present relations' classes to numerical indices mapping, where computation is performed on all train, dev and test samples taken together. Used in relation classification task.
        :direction2idx (dict): present relations' directions to numerical indices mapping, where computation is performed on all train, dev and test samples taken together. Used in relation classification task.
        :tags_corpus (flair.data.Corpus): flair.data.Corpus object for the tagging task ensured by the relex.models.EntitiesTagger module.
        :tag_type (str): tag type used by the relex.models.EntitiesTagger module.
        :tag_dict (flair.data.LabelDictionary): flair.data.LabelDictionary object for the tagging task ensured by the relex.models.EntitiesTagger module.
    """
    def save_tags_to_file(self, case, data):
        
        temp_path = "./tmp_tagger_corpus_files"
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        with open(temp_path+"/"+case+".txt", "w") as f:
            for sample in data:
                tokens = sample['sentence'].split()
                tags = ["<Out>"] * len(tokens)
                for i in range(len(tokens)):
                    if sample['head']['start'] <= i < sample['head']['end']:
                        if i == sample['head']['start']:
                            tags[i] = "<B-Entity>"
                        else:
                            tags[i] = "<I-Entity>"
                    elif sample['tail']['start'] <= i < sample['tail']['end']:
                        if i == sample['tail']['start']:
                            tags[i] = "<B-Entity>"
                        else:
                            tags[i] = "<I-Entity>"
                    else:
                        tags[i] = "<Out>"
                    f.write(tokens[i]+" "+tags[i]+"\n")
                f.write("\n")

    def make_tags_corpus(self):

        self.save_tags_to_file("train", self.train)
        self.save_tags_to_file("dev", self.dev)
        self.save_tags_to_file("test", self.test)
        columns = {0: 'text', 1: 'entities_tags'}
        tmp_path = "./tmp_tagger_corpus_files"
        corpus: Corpus = ColumnCorpus(tmp_path, columns, train_file='train.txt', test_file='test.txt', dev_file='dev.txt')
        for file in Path(tmp_path).glob('*.txt'):
            file.unlink()
        Path(tmp_path).rmdir()
        self.tags_corpus = corpus
        self.tag_type = 'entities_tags'
        self.tag_dict = corpus.make_label_dictionary(label_type=self.tag_type)

    def validate_sample(self, sample):
        try:
            assert isinstance(sample, dict)
            assert 'sentence' in sample and isinstance(sample['sentence'], str)
            assert 'class' in sample and isinstance(sample['class'], str)
            assert 'direction' in sample and isinstance(sample['direction'], str)
            assert 'head' in sample and isinstance(sample['head'], dict) and 'start' in sample['head'] and isinstance(sample['head']['start'], int) and 'end' in sample['head'] and isinstance(sample['head']['end'], int)
            assert 'tail' in sample and isinstance(sample['tail'], dict) and 'start' in sample['tail'] and isinstance(sample['tail']['start'], int) and 'end' in sample['tail'] and isinstance(sample['tail']['end'], int)
        except AssertionError:
            raise AssertionError("Invalid sample: {}".format(sample))

    def __init__(self, train=None, dev=None, test=None):
        """Initializes a RelexCorpus object with given data sets.
        
        Parameters:
            :train (list): training pre-annotated sentences with respect to relations.
            :dev (list): validation pre-annotated sentences with respect to relations.
            :test (list): test pre-annotated sentences with respect to relations.

        Returns:
            - nothing.
            - prints statistics about entites tags & relations types and directions to standard output.
        """
        
        try:
            assert isinstance(train, list) and len(train) > 0
            assert isinstance(dev, list) and len(dev) > 0
            assert isinstance(test, list) and len(test) > 0
        except AssertionError:
            raise ValueError("Invalid dataset format")

        for sample in train:
            self.validate_sample(sample)
        for sample in dev:
            self.validate_sample(sample)
        for sample in test:
            self.validate_sample(sample)

        self.class2idx = self.compute_class2idx(train + dev + test)
        self.direction2idx = self.compute_direction2idx(train + dev + test)
        
        self.train = RelexDataset(train)
        self.train_class_stats = self.get_per_class_count(train)
        self.train_direction_stats = self.get_per_direction_count(train)
        self.dev = RelexDataset(dev)
        self.dev_class_stats = self.get_per_class_count(dev)
        self.dev_direction_stats = self.get_per_direction_count(dev)
        self.test = RelexDataset(test)
        self.test_class_stats = self.get_per_class_count(test)
        self.test_direction_stats = self.get_per_direction_count(test)

        self.logger = self.init_logger()
        self.print_pairclf_stats()
        self.make_tags_corpus()
    
    def compute_class2idx(self, samples):
        class2idx = {}
        for sample in samples:
            if sample['class'] not in class2idx:
                class2idx[sample['class']] = len(class2idx)
        return class2idx
    
    def compute_direction2idx(self, samples):
        direction2idx = {}
        for sample in samples:
            if sample['direction'] not in direction2idx:
                direction2idx[sample['direction']] = len(direction2idx)
        return direction2idx
    
    def get_per_class_count(self, samples):
        class_count = {}
        for sample in samples:
            if sample['class'] not in class_count:
                class_count[sample['class']] = 0
            class_count[sample['class']] += 1
        return class_count
    
    def get_per_direction_count(self, samples):
        direction_count = {}
        for sample in samples:
            if sample['direction'] not in direction_count:
                direction_count[sample['direction']] = 0
            direction_count[sample['direction']] += 1
        return direction_count

    def init_logger(self):
        

        logger = logging.getLogger()
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        return logger

    def print_pairclf_stats(self):
        self.logger.info("\nRelation (type and/or direction) corpus details:")
        self.logger.info("------------------------------------------------")
        self.logger.info("Train samples: {}".format(len(self.train)))
        self.logger.info("Dev samples: {}".format(len(self.dev)))
        self.logger.info("Test samples: {}".format(len(self.test)))
        self.logger.info("Relation types: {}".format(len(self.class2idx)))
        self.logger.info("Train set stats: {}".format(self.train_class_stats))
        self.logger.info("Dev set stats: {}".format(self.dev_class_stats))
        self.logger.info("Test set stats: {}".format(self.test_class_stats))
        self.logger.info("Relation directions: {}".format(len(self.direction2idx)))
        self.logger.info("Train set stats: {}".format(self.train_direction_stats))
        self.logger.info("Dev set stats: {}".format(self.dev_direction_stats))
        self.logger.info("Test set stats: {}".format(self.test_direction_stats))
        self.logger.info("------------------------------------------------\n")
        self.logger.info("Entities tags corpus details:")
        self.logger.info("------------------------------------------------")
