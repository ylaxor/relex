from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger
from typing import Dict
import torch.nn as nn
from flair.data import Dictionary
from flair.data import Sentence

class EntitiesTagger(SequenceTagger):
    """EntitiesTagger class for predicting B.I.O-tags for single text tokens. Used to identify text entities that would form candidate pairs of potentially semantically related entities. Wraps a flair.models.SequenceTagger model, and can be parameterized by several parameters. Learning of this module can be easily achieved with the help of a relex.learners.EntitiesTaggerLearner instance, which will train, validate and test the entities tagger model weights on the given corpus and output optimal weights of the tagger to a reusable 'best-model.pt' checkpoint file. Output file of the learning process is a core flair.models.SequenceTagger checkpoint file.
    
    Parameters:
        :embeddings: stack of embedding.s to use during learning and prediction, wrapped inside a flair.data.StackedEmbeddings object.
        :tag_dictionary: flair.data.Dictionary containing all tags from corpus which can be predicted.
        :tag_type: type of tag which is going to be predicted in case a corpus has multiple annotations.
        :rnn_type: specifies the RNN type to use, default is 'LSTM', can choose between 'GRU' and 'RNN' as well.
        :hidden_size: hidden size of RNN layer.
        :rnn_layers: number of RNN layers.
        :bidirectional: if True, RNN becomes bidirectional.
        :use_crf: if True, use a Conditional Random Field for prediction, else linear map to tag space.
        :reproject_embeddings: if True, add a linear layer on top of embeddings, if you want to imitate
            fine tune non-trainable embeddings.
        :dropout: if > 0, then use dropout.
        :word_dropout: if > 0, then use word dropout.
        :locked_dropout: if > 0, then use locked dropout.
        :loss_weights: flair.data.Dictionary of weights for labels for the loss function
            (if any label's weight is unspecified it will default to 1.0).
        :allow_unk_predictions: if True, allow predictions for unknown words.
    """
    def __init__(
        self,
        embeddings: StackedEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        rnn_type: str = "LSTM",
        hidden_size: int = 256,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        use_crf: bool = True,
        reproject_embeddings: bool = True,
        dropout: float = 0.0,
        word_dropout: float = 0.05,
        locked_dropout: float = 0.5,
        loss_weights: Dict[str, float] = None,
        allow_unk_predictions: bool = False,
        **kwargs
    ):
        """Initialize a new EntitiesTagger model instance with the given parameters."""
        super(EntitiesTagger, self).__init__(embeddings=embeddings,tag_dictionary=tag_dictionary,tag_type=tag_type,rnn_type=rnn_type,hidden_size=hidden_size,rnn_layers=rnn_layers,bidirectional=bidirectional,use_crf=use_crf,reproject_embeddings=reproject_embeddings,dropout=dropout,word_dropout=word_dropout,locked_dropout=locked_dropout,loss_weights=loss_weights,allow_unk_predictions=allow_unk_predictions, **kwargs)
        pass

    @staticmethod
    def reload(base_path: str):
        """Reloads a pre-saved entities tagger model from the given folder, e.g after having preivously trained, validated, tested and saved a relex.models.EntitiesTagger model using the relex.learners.EntitiesTaggerLearner module.

        Parameters:
            :base_path: path to the folder containing the checkpoint file to use.
        
        Returns:
            :reloaded_model: A flair.models.SequenceTagger object containing the pre-saved tagger.
        """
        try:
            reloaded_model = SequenceTagger.load(base_path+'/best-model.pt')
            return reloaded_model
        except Exception as e:
            try:
                reloaded_model = SequenceTagger.load(base_path+'/final-model.pt')
                return reloaded_model
            except:
                print("Error while reloading tagging model from: {}".format(base_path))
                return None

    @staticmethod
    def infer(tagger: SequenceTagger, sentence: str):
        """Uses the given tagging model instance to infer tags for the given sentence.

        Parameters:
            :tagger: some learned flair.models.SequenceTagger model instance, e.g. reloaded after having trained, validated, tested and saved a relex.models.EntitiesTagger module using the relex.learners.EntitiesTaggerLearner module.
            :sentence: sentence to predict tags for.
        
        Returns:
            :tokens: list of tokens in the sentence, obtained by straightforward splitting the sentence on whitespace.
            :tags: list of tags for the given sentence.
        """
        try:
            sentence = Sentence(sentence)
            tagger.predict(sentence)
            tokens = [t.text for t in sentence]
            tags = [t.get_label(tagger.tag_type).value for t in sentence]
            return tokens, tags
        except Exception as e:
            print("Error while inferring tags for sentence: {}".format(e))
            return None
