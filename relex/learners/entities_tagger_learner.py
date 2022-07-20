from flair.trainers import ModelTrainer

class EntitiesTaggerLearner():
    """EntitiesTaggerLearner class to learn (train, validate, test) then save a model for tagging tokens that would form pairs of semantically related entites in a given text.
    Used to develop (learn) the entities tagging sub-module presented in relex.models.EntitiesTagger. Learning of the module, which leverages a BILSTM-CRF architecture, is done in a fully-supervised manner.

    Parameters:
        :entities_tagger: A relex.models.EntitiesTagger instance that will be trained.
        :corpus: A relex.data.RelexCorpus instance that will be used to train the entities tagger.
        :base_path: A string that represents the path where the model will be saved.
        :learning_rate: A float that represents the initial learning rate of the model.
        :mini_batch_size: An integer that represents the mini-batch size of the model.
        :max_epochs: An integer that represents the maximum number of learning epochs.
        :train_with_dev: A boolean that indicates if the model will be validated with the development set after each training epoch.
        :train_with_test: A boolean that indicates if the model will be trained with the test set after each training epoch.
        :main_evaluation_metric: A tuple that represents the main evaluation metric of the model. e.g ("macro avg", "f1-score")
        :anneal_factor: A float that represents the annealing factor of the learning rate used by some scheduler.
        :patience: An integer that represents the patience period of the scheduler the model.
        :save_final_model: A boolean that indicates if the model will be saved after the last epoch.
        :save_model_each_k_epochs: An integer that indicates if the model will be saved after each k epochs.
        :**kwargs: A dictionary that contains the other arguments of the falir.trainers.ModelTrainer class.

    """
    def __init__(
        self, 
        entities_tagger=None, 
        corpus=None, 
        base_path=None, 
        learning_rate=1e-1,
        mini_batch_size=16,
        max_epochs=75,
        train_with_dev=True,
        train_with_test=False,
        main_evaluation_metric= ("macro avg", "f1-score"),
        anneal_factor=0.5,
        patience=3,
        min_learning_rate=0.000001,
        save_final_model=True,
        save_model_each_k_epochs=0,
        **kwargs):
            self.important_args = locals()
            self.important_args.pop('self')
            self.important_args.pop('kwargs')
            self.important_args.pop('entities_tagger')
            self.important_args.pop('corpus')
            self.kwargs = kwargs
            self.trainer = ModelTrainer(entities_tagger, corpus.tags_corpus)

    def fit(self):
        """Learns the entities tagger sub-module on given corpus.

        Returns:
            Nothing. It shows the progress of learning process and the learned tagging sub-module is saved in the base_path.
        """
        self.trainer.train(**self.important_args, **self.kwargs)
