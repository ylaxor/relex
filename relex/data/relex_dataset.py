from torch.utils.data import Dataset

class RelexDataset(Dataset):
    """RelexDataset class to help validate and wrap the provided list of samples into a torch.utils.data.Dataset object.
    Samples in each param list must be dictionaries with a pre-defined (see mode parameter) subset of the following items:
                - 'sentence': a string containing the sentence.
                - 'class': a string containing the class of the relation.
                - 'direction': a string containing the direction of the relation.
                - 'head': a dictionary containing {'start': int, 'end': int} information of head entity.
                - 'tail': a dictionary containing {'start': int, 'end': int} information of tail entity.
        
    Parameters:
        :mode (str): should be either 'learning' or 'prediction'. In 'prediction' mode you only provide the 'sentence', 'head' and 'tail' information. In 'learning' mode, 'class' and 'direction' fields are additionally required. Defaults to 'learning'.
        :samples (list): a list of pre-annotated samples to wrap. 

    Attributes:
        :samples (list): a container list of validated pre-annotated samples.
        :mode (str): the selected loading mode.
    Raises:
        AssertionError: when one of the samples in the list is not a valid sample.
    """
    
    def validate_sample(self, sample):
        """Validates the given sample format and ensures it presents the necessary fields, according to 'self.mode' parameter.
        
        Parameters:
            :sample (dict): A single sample dictionary object to validate. 
        Raises:
            AssertionError: when one or more fields of the given sample are not validated.
        """
        try:
            assert isinstance(sample, dict)
            assert 'sentence' in sample and isinstance(sample['sentence'], str)
            if self.mode != "prediction":
                assert 'class' in sample and isinstance(sample['class'], str)
                assert 'direction' in sample and isinstance(sample['direction'], str)
            assert 'head' in sample and isinstance(sample['head'], dict) and 'start' in sample['head'] and isinstance(sample['head']['start'], int) and 'end' in sample['head'] and isinstance(sample['head']['end'], int)
            assert 'tail' in sample and isinstance(sample['tail'], dict) and 'start' in sample['tail'] and isinstance(sample['tail']['start'], int) and 'end' in sample['tail'] and isinstance(sample['tail']['end'], int)
        except AssertionError:
            raise AssertionError("Invalid sample: {}".format(sample))

    def __init__(self, samples=None, mode="learning"):
        """Initializes a torch.utils.data.Dataset object with the list of the provided (valid) samples.

        Parameters:
            :samples (list): a list of pre-annotated samples.
                Samples in the list must be dictionaries with the following keys:
                - 'sentence': a string containing the sentence.
                - 'class': a string containing the class of the relation.
                - 'direction': a string containing the direction of the relation.
                - 'head': a dictionary containing {'start': int, 'end': int} information of head entity.
                - 'tail': a dictionary containing {'start': int, 'end': int} information of tail entity.
            :mode (str): Should be either "learning" or "prediction". In "learning" mode, "class" and "direction" fields are additionally required. Defaults to "learning".
        
        Raises:
            AssertionError: when one of the samples in the list is not a valid sample.
        """
        assert mode in ['learning', 'prediction']
        self.mode = mode
        try: 
            assert isinstance(samples, list) and len(samples) > 0
        except AssertionError:
            raise AssertionError("Invalid dataset: {}".format(samples))
        for sample in samples:
            self.validate_sample(sample)
        self.samples = samples
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
