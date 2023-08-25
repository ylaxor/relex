As you might have understood, relex aims to emphasize modern deep-learning based approaches to tackle the problem of semantic relations extraction.
The package is therefore intended to provide necessary interfaces and routines for building custom pipelines of semantic relation extractions.
In the `Bringing together applied research and good coding practice <./philosophy.html#bringing-together-applied-research-and-good-coding-practice>`_ section of this documentation, we motivated how relex could evolve and be a more useful tool for people who want to build relation extraction systems: Even under the unified modular workflow, many effective approaches are still possible. 
Relex is made to make those accessible and directly applicable for its users personal needs (see `Contributing as applied (NLP) researcher <./contributions.html#contributing-as-applied-nlp-researcher>`_).
Below we will progressively give final usage examples of each effective pipeline incorporated in relex.

A first dummy approach
======================================

In this approach:

- Modular three-stage pipeline is respected.
- Learnable components are learned separately. 
- First stage: Tagging the input text with <B-Entity>, <I-Entity>, <Out> tags, where: <B-Entity> is the beginning of an entity implied in some relationship, <I-Entity> is used to tag the inside part.s of an entity, and <Out> to tag anything else (i.e any other text token).
- Second stage: Extracting the entities from the tagged text, and forming the list of all possible pairs (greedy pair formin).
- Third stage: Extracting the relations from the list of pairs, using a neural network classifier.
- Each learnable sub-modules leverages some specific transferred embedding models that help build rich text representations.
- Final components are gathered in a single (deterministic) routine: The predictor, whose role is to put everything together and predict the pairs, and the relations' types and/or directions between them, from the input text.

.. code-block:: python
   
    from relex.utilities import SeedEverything
    # Seed eveything for reproducibility
    SeedEverything(seed=1)

.. code-block:: python

    from relex.utilities import LoadFewRelData, LoadCustomCorpus
    # Import the FewRel dataset as example.
    data = LoadFewRelData(usage="demo-with-three-relations")
    # Or load your custom pre-annotated data samples.
    custom_train_data = LoadCustomCorpus(file_path="./demo-custom-corpus.jsonl")
    custom_dev_data = LoadCustomCorpus(file_path="./demo-custom-corpus.jsonl")
    custom_test_data = LoadCustomCorpus(file_path="./demo-custom-corpus.jsonl")
    
An example of valid custom corpus file can be found in the `following link <https://github.com/ylaxor/relex/blob/main/demo-custom-corpus.jsonl>`_.
Such files can be created using awesome GUI annotation tools like: `doccano <https://github.com/doccano/doccano>`_.

.. code-block:: python

    from relex.data import RelexCorpus
    # Wrap your data into a RelexCorpus object for later learning purposes.
    corpus = RelexCorpus(train=data["train"], dev=data["dev"], test=data["test"])
    #corpus = RelexCorpus(train=custom_train_data, dev=custom_dev_data, test=custom_test_data)



.. code-block:: python

    from relex.utilities import EmbeddingsLoader
    # Pick some embeddings to use for each sub-module.
    # Choice of embeddings for the EntitiesTagger sub-module.
    tagger_embeddings = EmbeddingsLoader(requested_embeddings={"glove":{"family":"Word"}})
    # Choice of embeddings for the PairClassifier sub-module.
    pair_clf_embeddings = EmbeddingsLoader(requested_embeddings={"bert-base-uncased":{"family":"Transformer"}})

.. code-block:: python

    from relex.models import EntitiesTagger, PairClassifier
    # Instantiate and configure the sub-modules before proceeding to training.
    # Entities tagging sub-module: configure the model.
    tagger_model = EntitiesTagger(
        embeddings=tagger_embeddings, 
        tag_dictionary=corpus.tag_dict, 
        tag_type=corpus.tag_type,
        hidden_size=100,
        reproject_embeddings=True,
        loss_weights=None,
        allow_unk_predictions=True
        )
    # Pair classification sub-module: configure the model.
    pair_clf_model = PairClassifier(
        embeddings=pair_clf_embeddings,
        use_specific_embeddings=True,
        reproject_embeddings=True,
        projection_size=100,
        hidden_size=100,
        nb_classes=len(corpus.class2idx),
        nb_directions=len(corpus.direction2idx),
        pair_pooling_operation="stack",
        init_class_alpha=1.0,
        init_direction_alpha=1.0,
        )
    
.. code-block:: python

    from relex.learners import EntitiesTaggerLearner, PairClassifierLearner
    # Setup learning loops parameters for each sub-module, and launch them separately.
    # Setup the learning loop for the entities tagging sub-module.
    tagger_learner = EntitiesTaggerLearner(
        entities_tagger=tagger_model, 
        corpus=corpus, 
        base_path="./resources/entities-tagging", 
        max_epochs=5,
        learning_rate=1e-1,
        mini_batch_size=16,
        patience=1,
        anneal_factor=0.5,
        main_evaluation_metric=("macro avg", "f1-score"),
        )
    # Setup the learning loop for the pair classification sub-module.
    pair_clf_learner = PairClassifierLearner(
        pair_classifier=pair_clf_model,
        target_task_name="relation-type-only",
        loss_aggregation_mode="off",
        tune_aggregation_weights=False,
        corpus=corpus,
        base_path="./resources/pair-classification",
        max_epochs=5,
        learning_rate=1e-1,
        mini_batch_size=16,
        patience=1,
        anneal_factor=0.5,
        eval_metric_type="macro",
        )
    # Launch the learning loops in separate fashion.
    tagger_learner.fit()
    pair_clf_learner.fit()

.. code-block:: python

    from relex.predictors import GreedyPredictor
    #reload best learned sub-modules, namely the entities tagger and pairs classifier.
    best_tagger = EntitiesTagger.reload("./resources/entities-tagging")
    best_pair_clf = PairClassifier.reload("./resources/pair-classification/relation-type-only")
    #plug them into a greedy (with brutal pair forming strategy) predictor.
    extractor = GreedyPredictor(
        tagger=best_tagger,
        classifier=best_pair_clf,
        class2idx=corpus.class2idx,
        direction2idx=corpus.direction2idx,
    )
    #re-extract relation information on some sample from the test test
    test_sample = next(iter(corpus.test))
    print(test_sample)
    test_raw_text = test_sample["sentence"]
    extractor.predict(test_raw_text)
