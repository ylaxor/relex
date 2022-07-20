from ..models import EntitiesTagger
from ..models import PairClassifier
from ..data import RelexDataset
from flair.data import Sentence
from itertools import product as product
from torch.utils.data import DataLoader


class GreedyPredictor:
    """GreedyPredictor class to wrap previously trained tagging and classification sub-modules, along with a greedy approach to the sub-task of forming candidate pairs.
    Used as the final relation extractor.

    Parameters:
        :tagger: A reloaded (already learned) relex.models.EntitiesTagger instance.
        :classifier: A reloaded (already learned) relex.models.PairClassifier instance.
        :class2idx: A dictionary that maps the relation types to their indices. Must be the same as the one used for training the classifier.
        :direction2idx: A dictionary that maps the relation direction names to their indices. Must be the same as the one used for training the classifier.
    """
    def __init__(self, tagger, classifier, class2idx, direction2idx):
        try:
            assert isinstance(tagger, EntitiesTagger.__bases__[0])
            assert isinstance(classifier, PairClassifier)
            assert isinstance(class2idx, dict)
            assert isinstance(direction2idx, dict)
        except AssertionError:
            raise ValueError("Tagger and classifier must be of type EntitiesTagger and PairClassifier")
        self.tagger = tagger
        self.classifier = classifier
        self.class2idx = class2idx
        self.direction2idx = direction2idx
    
    def combine_pairs(self, L, tokens):
        L = sorted(L, key=lambda x: int(x["start"]))
        combos = []
        while len(L):
            head = L.pop(0)
            combos.extend(list(product([head], L)))
        pairs = []
        for combo in combos:
            sample = {"sentence": " ".join(tokens), "head": combo[0], "tail": combo[1]}
            pairs.append(sample)
        return pairs

    def extract_entities(self, tokens, tags):
        entities_record = []
        for i, token, tag in zip(range(len(tokens)), tokens, tags):
            entity = []
            if "<B-" in tag:
                entity.append(token)
                start = i
                end = i+1
                for delta in range(1, len(tokens)-start):
                    if "<I-" not in tags[start+delta]:
                        break
                    else:
                        entity.append(tokens[start+delta])
                        end = start+delta+1
                entities_record.append({"start":start, "end":end})
        return entities_record

    def predict(self, sentences):
        """Extracts the effective relations between pairs of found entities in a given collection sentence.

        Parameters:
            :sentences: A list of strings that represents the sentences to be tagged.
        Returns:
            :predictions: A list of dictionaries that contains the predicted relations between formed pairs of found entities for each sentence.
        """

        if isinstance(sentences, str):
            sentences = [sentences]
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        predictions = []
        for sentence in sentences:
            
            s = Sentence(sentence.split())
            self.tagger.predict(s)
            s_tokens = [token.text for token in s]
            s_tags = [token.get_label('entities_tags').value for token in s]
            s_entities = self.extract_entities(s_tokens, s_tags)
            pairs = self.combine_pairs(s_entities, s_tokens)
            if len(pairs) > 0:
                prediction = {}
                
                prediction["pairs"] = pairs
                pairs = RelexDataset(samples=pairs, mode="prediction")
                pairs_loader = next(iter(DataLoader(pairs, batch_size=len(pairs), shuffle=False, num_workers=0)))
                batch_logits_class, batch_logits_direction = self.classifier(pairs_loader)
                idx2class = {idx:class_name for class_name, idx in self.class2idx.items()}
                idx2direction = {idx:direction_name for direction_name, idx in self.direction2idx.items()}
                classes = [idx2class[idx.item()] for idx in batch_logits_class.argmax(dim=1)]
                directions = [idx2direction[idx.item()] for idx in batch_logits_direction.argmax(dim=1)]
                prediction = {"tokens": s_tokens, "tags": s_tags, "pairs":pairs,"classes": classes, "directions": directions}
                predictions.append(prediction)
            else:
                predictions.append({"tokens": s_tokens, "tags": s_tags, "pairs": [], "classes": [], "directions": []})
        return predictions