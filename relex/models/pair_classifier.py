import torch
import torch.nn as nn
from flair.data import Sentence
import pickle 
from ..utilities.utils import GetAvailableDevice
import logging
import sys
from flair.embeddings import StackedEmbeddings, CharacterEmbeddings

class PairClassifier(nn.Module):
    """PairClassifier class for classifying relation type and/or direction for a given candidate pair of text entities. Used to predict relation nature (type, direction, or jointly both) between candidate pairs, e.g those that would be formed thanks to the entities tags inferred by a pre-trained relex.models.EntitiesTagger module. Can be parameterized by several parameters.

    Parameters:
        :embeddings: stack of embedding.s to use during learning and prediction, wrapped inside a flair.data.StackedEmbeddings object.
        :use_specific_embeddings: if True, a specific (task-trainable) char-level-based word embeddings layer is added to the stack of external embeddings. Its role is to help consolidate (since trainable) the classifier's knowledge about relations types and/or directions.
        :pair_pooling_operation: "average" or "stack" to use for pooling the features of the two entities of a pair.
        :reproject_embeddings: if True, the obtained pair representation is projected on one extra hidden layer before being fed to the main hidden layer.
        :projection_size: the size of the extra pair-projection layer. Used when reproject_embeddings is True.
        :hidden_size: hidden size of main linear layer used to classify pairs' representations (or projected version of them if reproject_embeddings is True).
        :nb_classes: number of relation types to predict. E.g 3 for "Has nationality", "Located in", and "Plays".
        :nb_directions: number of relation directions to predict. E.g 2 for "left-to-right" and "right-to-left".
        :init_class_alpha: (Optional) initial value of some coefficient to multiply with the relation type's loss during learning of the model, and when combined loss aggregation mode is active.
        :init_direction_alpha: (Optional) initial value of some coefficient to multiply with the relation direction's loss during learning of the model, and when combined loss aggregation mode is active.
    """
    
    def __init__(self, embeddings=None, reproject_embeddings=True, projection_size=256, hidden_size=256, nb_classes=2, nb_directions=2, use_specific_embeddings=True, pair_pooling_operation="stack", init_class_alpha=1.0, init_direction_alpha=1.0):
        """Initialize a new PairClassifier instance with the given parameters."""
        super().__init__()
        try:
            assert isinstance(embeddings, StackedEmbeddings)
            assert isinstance(use_specific_embeddings, bool)
            assert isinstance(pair_pooling_operation, str) and pair_pooling_operation in ["stack", "average"]
            assert isinstance(reproject_embeddings, bool)
            assert isinstance(projection_size, int) and projection_size > 0
            assert isinstance(hidden_size, int) and hidden_size > 0
            assert isinstance(nb_classes, int) and nb_classes > 0
            assert isinstance(nb_directions, int) and nb_directions > 0
            assert isinstance(init_class_alpha, float)
            assert isinstance(init_direction_alpha, float)
        except AssertionError:
            raise Exception("Invalid parameters")

        self.device = GetAvailableDevice()
        self.pair_pooling_operation = pair_pooling_operation
        self.use_projection = reproject_embeddings
        self.projection_dim = projection_size
        self.hidden_dim = hidden_size
        self.use_specific_embeddings = use_specific_embeddings
        self.external_embeddings = embeddings
        self.act_hidden = nn.ReLU()
        self.act_out = nn.Softmax(dim=0)
        self.embedding_layer = self.stack_embeddings(embeddings, use_specific_embeddings)
        self.alpha_class = nn.Parameter(torch.Tensor([init_class_alpha]))
        self.alpha_direction = nn.Parameter(torch.Tensor([init_direction_alpha]))
        if reproject_embeddings:
            self.features_projector = nn.Linear(self.compute_pair_features_vector_dim(), projection_size)
            self.bn_projection = nn.BatchNorm1d(projection_size)
            self.fully_connected_first = nn.Linear(projection_size, hidden_size)
            self.bn_first = nn.BatchNorm1d(hidden_size)
            self.fully_connected_class = nn.Linear(hidden_size, nb_classes)
            self.fully_connected_direction = nn.Linear(hidden_size, nb_directions)
        else:
            self.fully_connected_first = nn.Linear(self.compute_pair_features_vector_dim(), hidden_size)
            self.bn_first = nn.BatchNorm1d(hidden_size)
            self.fully_connected_class = nn.Linear(hidden_size, nb_classes)
            self.fully_connected_direction = nn.Linear(hidden_size, nb_directions)
 
    def stack_embeddings(self, external_embeddings, use_specific_embeddings):
        if use_specific_embeddings:
            full_embeddings = [CharacterEmbeddings()]
        else:
            full_embeddings = []
            full_embeddings.extend(external_embeddings)
        return StackedEmbeddings(full_embeddings)
    
    def compute_pair_features_vector_dim(self):
        entity_dim = self.embedding_layer.embedding_length 
        pair_dim = entity_dim if self.pair_pooling_operation == "average" else 2 * entity_dim
        return pair_dim

    def pool_enntity_embeddings(self, entity_embeddings):
        if len(entity_embeddings) == 1:
            return entity_embeddings[0]
        else:
            return torch.mean(torch.stack([entity_embeddings[0], entity_embeddings[-1]]), dim=0)
            
    def pool_pair_entities_features(self, head_features, tail_features):
        if self.pair_pooling_operation == "average":
            return torch.mean(torch.stack([head_features, tail_features]), dim=0)
        if self.pair_pooling_operation == "stack":
            return torch.cat([head_features, tail_features])

    def get_pair_features(self, sample):
        tokens = sample["sentence"].split(" ")
        sentence = Sentence(tokens)
        head = sample["head"]
        tail = sample["tail"]
        self.embedding_layer.embed(sentence)
        embeddings = list(map(lambda token: token.embedding, sentence))
        head_embeddings = embeddings[head["start"]:head["end"]]
        tail_embeddings = embeddings[tail["start"]:tail["end"]]
        head_features = self.pool_enntity_embeddings(head_embeddings)
        tail_features = self.pool_enntity_embeddings(tail_embeddings)
        pair_features = self.pool_pair_entities_features(head_features, tail_features)
        return pair_features

 
    @staticmethod
    def reload(base_path: str):
        """Reloads a pre-saved relex.models.PairClassifier model from the given folder, e.g after having preivously learned, tested and saved such a model using the relex.learners.PairClassifierLearner module.

        Parameters:
            :base_path: path to the folder containing the 'best-model.pt' checkpoint file to use.
        
        Returns:
            :reloaded_model: A relex.models.PairClassifier instance reloaded from the given file.
        """

        try:
            logger = logging.getLogger()
            if (logger.hasHandlers()):
                logger.handlers.clear()
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s | %(message)s')
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)
            logger.addHandler(stdout_handler)
            logger.info("loading file {} configured using {}".format(base_path+"/best-model.pt", base_path+"/model.cfg"))
            with open(base_path+"/model.cfg", "rb") as f:
                model = pickle.load(f)
            model.load_state_dict(torch.load(base_path+"/best-model.pt"))
            model.eval()
            return model
        except:
            print("Error loading model")
            return None
    
    @staticmethod
    def infer(pair_classifier, batch_samples):
        """Uses the given pair classifier instance to perform forward pass on the given batch of samples and returns the predicted relation type and direction for each.
        Samples in the batch must be dictionaries with the following items:
            - 'sentence': a string containing the sentence.
            - 'head': a dictionary containing {'start': int, 'end': int} information of head entity. The 'start' and 'end' are indices of the first and last token of the head (left) entity within the ordered list of tokens in the sentence.
            - 'tail': a dictionary containing {'start': int, 'end': int} information of tail entity. The 'start' and 'end' are indices of the first and last token of the tail (right) entity within the ordered list of tokens in the sentence.


        Parameters:
            :pair_classifier: some learned relex.models.PairClassifier model instance, eg. reloaded after having trained, validated, tested and saved a relex.models.PairClassifier module using the relex.learners.PairClassifierLearner module.
            :batch_samples: A torch-based batch object of size N samples, e.g through next(iter(torch.utils.data.DataLoader(RelexDataset, batch_size=N, shuffle=True))

        Returns:
            :predicted_relation_type: A list of predicted relation types for the given samples.
            :predicted_relation_direction: A list of predicted relation directions for the given samples.
        """
        with torch.no_grad():
            batch_logits_class, batch_logits_direction = pair_classifier.forward(batch_samples)
            predicted_relation_type = torch.argmax(batch_logits_class, dim=1)
            predicted_relation_direction = torch.argmax(batch_logits_direction, dim=1)
            return predicted_relation_type, predicted_relation_direction
    
    def forward(self, batch_samples):
        """Performs forward pass - for both relation type and direction - on the N samples in the given batch. In the relex.learners.PairClassifierLearner module, you can select which of relation's (type, direction) to train for, and only the targeted logits (relation type, direction, or jointly both) are used to compute the loss.es and update only the branch of network weights being invoked accordingly.
        Samples in the batch must be dictionaries with the following items:
            - 'sentence': a string containing the sentence.
            - 'head': a dictionary containing {'start': int, 'end': int} information of head entity.
            - 'tail': a dictionary containing {'start': int, 'end': int} information of tail entity.
        
        Parameters:
            :batch_samples: A torch-based batch object of size N samples, e.g through next(iter(torch.utils.data.DataLoader(RelexDataset, batch_size=N, shuffle=True))

        Returns:
            :batch_logits_relation_type: (N, number of relation types) e.g (N, 2) for two relation types, where N is the number of samples in the batch.
            :batch_logits_relation_direction: (N, number of relation directions) e.g (N, 2) for two relation directions, where N is the number of samples in the batch.
        """
        batch_size = len(batch_samples["sentence"])
        batch_pair_features = [self.get_pair_features({"sentence":batch_samples["sentence"][i],"head":{"start":batch_samples["head"]["start"][i],"end":batch_samples["head"]["end"][i]}, "tail":{"start":batch_samples["tail"]["start"][i],"end":batch_samples["tail"]["end"][i]}}) for i in range(batch_size)]
        batch_pair_features = torch.stack(batch_pair_features)
        batch_pair_features = batch_pair_features.to(self.device)
        if self.use_projection:
            batch_h1 = self.features_projector(batch_pair_features)
            batch_h1 = self.bn_projection(batch_h1)
            batch_h1 = self.act_hidden(batch_h1)
            batch_h2 = self.fully_connected_first(batch_h1)
            batch_h2 = self.bn_first(batch_h2)
            batch_h2 = self.act_hidden(batch_h2)
            batch_out_class = self.fully_connected_class(batch_h2)
            batch_out_direction = self.fully_connected_direction(batch_h2)
            batch_logits_class = self.act_out(batch_out_class)
            batch_logits_direction = self.act_out(batch_out_direction) 
        else:
            batch_h1 = self.fully_connected_first(batch_pair_features)
            batch_h1 = self.bn_first(batch_h1)
            batch_h1 = self.act_hidden(batch_h1)
            batch_out_class = self.fully_connected_class(batch_h1)
            batch_out_direction = self.fully_connected_direction(batch_h1)
            batch_logits_class = self.act_out(batch_out_class)
            batch_logits_direction = self.act_out(batch_out_direction)   
        return batch_logits_class, batch_logits_direction
