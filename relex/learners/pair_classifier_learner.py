import os
import sys
import time
import random
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics import F1Score
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from ..models import PairClassifier
from ..data import RelexCorpus
import pickle 
from ..utilities.utils import GetAvailableDevice

class PairClassifierLearner:
    """PairClassifierLearner class to learn (train, validate, test) then save a model for classifying relation types and/or directions for previously formed pairs of entities.
    Used to develop (learn) the pairs classification sub-module present in relex.models.PairClassifier. 
    Learning of the module, which first forms a generic representation of the pair of entities (thanks to some transfered embeddings) then feeds them to a classifier network in order to predict desired information, is again performed in a fully-supervised manner, using the same learning corpus as the one that would be used to develop the tagging module.

    Parameters:
        :pair_classifier: PairClassifier object, which is the model to be learned.
        :corpus: RelexCorpus object, which is the corpus used to develop the module.
        :target_task_name: str, which is the name of the task to be learned. Can be "relation-type-only", "relation-direction-only", "relation-type-and-direction".
        :loss_aggregation_mode: str, which is the mode of aggregation of the loss function. Can be "average", "sum", "weighted-sum", "pick-one", "off".
        :tune_aggregation_weights: bool, which is True if the weights of the aggregation function should be tuned, False otherwise. Active only if target_task_name is "relation-type-and-direction".
        :max_epochs: int, which is the maximum number of epochs to be performed during the learning process.
        :mini_batch_size: int, which is the size of the mini-batches used during the learning process.
        :learning_rate: float, which is the initial learning rate used during the learning process.
        :patience: int, which is the number of epochs to be performed before the learning rate is decreased if the validation score does not improve.
        :max_bad_epochs: int, which is the maximum number of epochs to be performed before the learning process is stopped if the validation score does not improve.
        :anneal_factor: float, which is the factor by which the learning rate is decreased.
        :eval_metric_type: str, which is the type of evaluation metric to be used. Can be "macro", "micro", f-1 scores.
        :base_path: str, which is the path to the folder where the output model files will be saved.
        :log_to_stdout: bool, which is True if the learning logs should be printed to stdout, False otherwise.
        :log_history_plots: bool, which is True if the learning history plots should be printed, False otherwise.
        :display_plots: bool, which is True if the learning history plots should be displayed, False otherwise.
        :verbose_training: bool, which is True if the training logs should be printed, False otherwise.
        :verbosity_period: int, which is the number of batches between each verbose print.
    """

    def __init__(
        self, 
        pair_classifier=None, 
        corpus=None, 
        target_task_name="relation-type-only", 
        loss_aggregation_mode="off", 
        tune_aggregation_weights=False, 
        max_epochs=75, 
        mini_batch_size=16,
        learning_rate=1e-1, 
        patience=4, 
        anneal_factor=0.5, 
        max_bad_epochs=12, 
        eval_metric_type="macro", 
        base_path=None, 
        save_final_model=True, 
        log_to_stdout=True, 
        log_history_plots=True, 
        display_plots=True, 
        verbose_training=True, 
        verbosity_period=25):
            try:
                assert target_task_name in ["relation-type-only", "relation-direction-only", "relation-type-and-direction"]
                assert loss_aggregation_mode in ["average", "sum", "weighted-sum", "pick-one", "off"]
                if target_task_name in ["relation-type-only", "relation-direction-only"]:
                    assert loss_aggregation_mode == "off"
                
                assert isinstance(tune_aggregation_weights, bool)
                assert isinstance(max_epochs, int) and max_epochs > 0
                assert isinstance(mini_batch_size, int) and mini_batch_size > 0
                assert isinstance(learning_rate, float) and learning_rate > 0
                assert isinstance(patience, int) and patience > 0
                assert isinstance(anneal_factor, float) and anneal_factor > 0
                assert isinstance(max_bad_epochs, int) and max_bad_epochs > 0
                assert eval_metric_type in ["macro", "micro"]
                assert isinstance(pair_classifier, PairClassifier)
                assert isinstance(corpus, RelexCorpus)
                assert isinstance(base_path, str)
                assert isinstance(save_final_model, bool)
                assert isinstance(log_to_stdout, bool)
                assert isinstance(log_history_plots, bool)
                assert isinstance(display_plots, bool)
                assert isinstance(verbose_training, bool)
                assert isinstance(verbosity_period, int) and verbosity_period > 0
            except AssertionError:
                raise ValueError("Invalid input parameters")
                
            self.model = pair_classifier
            self.corpus = corpus
            self.target_task_name = target_task_name
            self.loss_aggregation_mode = loss_aggregation_mode
            self.tune_aggregation_weights = tune_aggregation_weights
            self.max_epochs = max_epochs
            self.initial_learning_rate = learning_rate
            self.decay_patience = patience
            self.decay_factor = anneal_factor
            self.max_bad_epochs = max_bad_epochs
            self.eval_metric_type = eval_metric_type
            self.available_device = GetAvailableDevice()
            self.output_folder = base_path
            self.save_final_model = save_final_model
            self.log_to_stdout = log_to_stdout
            self.log_history_plots = log_history_plots
            self.display_plots = display_plots
            self.verbose_training = verbose_training
            self.verbosity_period = verbosity_period
            self.batch_size = mini_batch_size
            self.model.to(self.available_device)
            if not self.tune_aggregation_weights:
                self.model.alpha_class.requires_grad = False
                self.model.alpha_direction.requires_grad = False
        
    def get_class_fields_dict(self):
        return {
            "relation-type-only": ["relation_type"],
            "relation-direction-only": ["relation_direction"],
            "relation-class-and-direction": ["relation_type", "relation_direction"]
        }
    
    def init_logger(self):
        Path(self.output_folder+"/"+self.target_task_name).mkdir(parents=True, exist_ok=True)
        try:
            os.remove(self.output_folder+"/"+self.target_task_name+"/learning.log")
        except OSError:
            pass
        logger = logging.getLogger()
        if (logger.hasHandlers()):
            logger.handlers.clear()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        if self.log_to_stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)
            logger.addHandler(stdout_handler)
        file_handler = logging.FileHandler(self.output_folder+"/"+self.target_task_name+'/learning.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def regenerate_empty_output_directory(self):
        Path(self.output_folder+"/"+self.target_task_name).mkdir(parents=True, exist_ok=True)        
        try:
            final_exists = os.path.isfile(self.output_folder+"/"+self.target_task_name+"/final-model.pt")
            best_exists = os.path.isfile(self.output_folder+"/"+self.target_task_name+"/best-model.pt")
            if final_exists:
                os.remove(self.output_folder+"/"+self.target_task_name+"/final-model.pt")
            if best_exists:
                os.remove(self.output_folder+"/"+self.target_task_name+"/best-model.pt")
            return "Old files were found and removed from '{}'".format(self.output_folder+"/"+self.target_task_name)
        except OSError:
            return "No old files found at '{}'".format(self.output_folder+"/"+self.target_task_name)
    
    def get_data_loaders(self):
        train_dataloader = DataLoader(self.corpus.train, batch_size=self.batch_size, shuffle=True)
        dev_dataloader = DataLoader(self.corpus.dev, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.corpus.test, batch_size=self.batch_size, shuffle=True)
        return train_dataloader, dev_dataloader, test_dataloader
    
    def compute_total_loss(self, loss_mode, class_loss, direction_loss, a_class, a_direction):
        if loss_mode == "average":
            return (class_loss + direction_loss) / 2
        if loss_mode == "weighted-sum":
            return (class_loss * a_class + direction_loss * a_direction)
        if loss_mode == "sum":
            return class_loss + direction_loss
        if loss_mode == "pick-one":
            return random.choice([class_loss, direction_loss])
        if loss_mode == "off":
            target_task = self.target_task_name
            if "relation" in target_task:
                return class_loss
            if "direction" in target_task:
                return direction_loss

    def learn_step(self, mode, data_loader, criterion, optimizer, epoch, logger, eval_metrics):

        if mode == "train":
            self.model.train()
        elif mode == "dev":
            self.model.eval()
            eval_metric_class = eval_metrics["class"]
            eval_metric_direction = eval_metrics["direction"]
        
        Class_loss = 0.0
        Direction_loss = 0.0
        Total_loss = 0.0

        Class_score = 0.0
        Direction_score = 0.0
        Total_score = 0.0

        for k, batch_sample in enumerate(data_loader):
            
            predicted_relation_class, predicted_relation_direction = self.model(batch_sample)
            
            if self.target_task_name == "relation-type-only":
                
                true_relation_class = torch.tensor([self.corpus.class2idx[c] for c in batch_sample["class"]])
                true_relation_class = true_relation_class.to(self.available_device)
                
                class_loss = criterion(input=predicted_relation_class, target=true_relation_class)
                direction_loss = torch.tensor(0.0, requires_grad=True, device=self.available_device)
                total_loss = self.compute_total_loss(self.loss_aggregation_mode, class_loss, direction_loss, None, None)

                Class_loss += class_loss.item() 
                Direction_loss += direction_loss.item()
                Total_loss += total_loss.item()

                if mode == "dev":
                    cls_score = eval_metric_class(predicted_relation_class, true_relation_class)
                    Class_score += cls_score.item()
                    Direction_score += float(0.0)
                    Total_score += cls_score.item()

            elif self.target_task_name == "relation-direction-only":

                true_relation_direction = torch.tensor([self.corpus.direction2idx[d] for d in batch_sample["direction"]])
                true_relation_direction = true_relation_direction.to(self.available_device)
                
                direction_loss = criterion(input=predicted_relation_direction, target=true_relation_direction)
                class_loss = torch.tensor(0.0, requires_grad=True, device=self.available_device)
                total_loss = self.compute_total_loss(self.loss_aggregation_mode, class_loss, direction_loss, None, None)
               
                Class_loss += class_loss.item()
                Direction_loss += direction_loss.item()
                Total_loss += total_loss.item()

                if mode == "dev":
                    Class_score += float(0.0)
                    dir_score = eval_metric_direction(predicted_relation_direction, true_relation_direction)
                    Direction_score += dir_score.item()
                    Total_score += dir_score.item()

            elif self.target_task_name == "relation-type-and-direction":

                true_relation_class = torch.tensor([self.corpus.class2idx[c] for c in batch_sample["class"]])
                true_relation_class = true_relation_class.to(self.available_device)
                
                true_relation_direction = torch.tensor([self.corpus.direction2idx[d] for d in batch_sample["direction"]])
                true_relation_direction = true_relation_direction.to(self.available_device)
                
                class_loss = criterion(input=predicted_relation_class, target=true_relation_class)
                direction_loss = criterion(input=predicted_relation_direction, target=true_relation_direction)
                total_loss = self.compute_total_loss(self.loss_aggregation_mode, class_loss, direction_loss, self.model.alpha_class, self.model.alpha_direction)
                
                Class_loss += class_loss.item()
                Direction_loss += direction_loss.item()
                Total_loss += total_loss.item()
                
                if mode == "dev":
                    cls_score = eval_metric_class(predicted_relation_class, true_relation_class)
                    dir_score = eval_metric_direction(predicted_relation_direction, true_relation_direction)
                    Class_score += cls_score.item()
                    Direction_score += dir_score.item()
                    Total_score += (cls_score.item() + dir_score.item()) / 2
            else:
                raise ValueError("Unknown mode: {}".format(self.target_task_name))

            if mode == "train":
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if mode == "train" and self.verbose_training and (k+1) % self.verbosity_period == 0:
                logger.info("Epoch {:05d}/{:05d}: batch {:05d}/{:05d}: loss: {:.6f}".format(epoch+1, self.max_epochs, k+1, len(data_loader), total_loss.item()))
            
        Class_loss /= len(data_loader)
        Direction_loss /= len(data_loader)
        Total_loss /= len(data_loader)

        if mode == "dev":
            Class_score /= len(data_loader)
            Direction_score /= len(data_loader)
            Total_score /= len(data_loader)
            return Class_loss, Direction_loss, Total_loss, Class_score, Direction_score, Total_score
        else:
            return Class_loss, Direction_loss, Total_loss

    def test_step(self, logger, test_loader, model):
        best_model = model
        best_model.load_state_dict(torch.load(self.output_folder+"/"+self.target_task_name+"/best-model.pt"))
        best_model.to(self.available_device)
        best_model.eval()
        logger.info("Finished loading best-model from '{}'".format(self.output_folder+"/"+self.target_task_name+"/best-model.pt"))
        
        test_true_class_labels = []
        test_pred_class_labels = []
        test_true_direction_labels = []
        test_pred_direction_labels = []

        for batch_sample in test_loader:
            predicted_relation_class, predicted_relation_direction = best_model(batch_sample)
            if self.target_task_name == "relation-type-only":
                true_relation_class = [self.corpus.class2idx[c] for c in batch_sample["class"]]
                predicted_relation_class = torch.argmax(predicted_relation_class, dim=1).tolist()
                test_true_class_labels.extend(true_relation_class)
                test_pred_class_labels.extend(predicted_relation_class)
                test_true_direction_labels.append(None)
                test_pred_direction_labels.append(None)
            elif self.target_task_name == "relation-direction-only":
                true_relation_direction = [self.corpus.direction2idx[d] for d in batch_sample["direction"]]
                predicted_relation_direction = torch.argmax(predicted_relation_direction, dim=1).tolist()
                test_true_class_labels.append(None)
                test_pred_class_labels.append(None)
                test_true_direction_labels.extend(true_relation_direction)
                test_pred_direction_labels.extend(predicted_relation_direction)
            elif self.target_task_name == "relation-type-and-direction":
                true_relation_class = [self.corpus.class2idx[c] for c in batch_sample["class"]]
                true_relation_direction = [self.corpus.direction2idx[d] for d in batch_sample["direction"]]
                predicted_relation_class = torch.argmax(predicted_relation_class, dim=1).tolist()
                predicted_relation_direction = torch.argmax(predicted_relation_direction, dim=1).tolist()
                test_true_class_labels.extend(true_relation_class)
                test_pred_class_labels.extend(predicted_relation_class)
                test_true_direction_labels.extend(true_relation_direction)
                test_pred_direction_labels.extend(predicted_relation_direction)
        class_target_names = list(self.corpus.class2idx.keys())
        class_target_labels = list(self.corpus.class2idx.values())
        direction_target_names = list(self.corpus.direction2idx.keys())
        direction_target_labels = list(self.corpus.direction2idx.values())
        class_test_score = []
        direction_test_score = []
        total_test_score = []
        if self.target_task_name == "relation-type-only":
            logger.info("Rel. Type f1-scores on test samples:\n\n"+str(classification_report(test_true_class_labels, test_pred_class_labels, target_names=class_target_names, labels=class_target_labels, zero_division=0)))
            class_precision,class_recall,class_fscore,class_support=score(test_true_class_labels,test_pred_class_labels,average=self.eval_metric_type, zero_division=0)
            class_test_score.append(class_fscore)
            total_test_score.append(class_fscore)
            direction_test_score.append(0)
        elif self.target_task_name == "relation-direction-only":
            logger.info("Rel. Direction f1-scores on test samples:\n\n"+str(classification_report(test_true_direction_labels, test_pred_direction_labels, target_names=direction_target_names, labels=direction_target_labels, zero_division=0)))
            direction_precision,direction_recall,direction_fscore,direction_support=score(test_true_direction_labels,test_pred_direction_labels,average=self.eval_metric_type, zero_division=0)
            direction_test_score.append(direction_fscore)
            total_test_score.append(direction_fscore)
            class_test_score.append(0)
        elif self.target_task_name == "relation-type-and-direction":
            logger.info("Rel. Type f1-scores on test samples:\n\n"+str(classification_report(test_true_class_labels, test_pred_class_labels, target_names=class_target_names, labels=class_target_labels, zero_division=0)))
            logger.info("Rel. Direction f1-scores on test samples:\n"+str(classification_report(test_true_direction_labels, test_pred_direction_labels, target_names=direction_target_names, labels=direction_target_labels, zero_division=0)))
            class_precision,class_recall,class_fscore,class_support=score(test_true_class_labels,test_pred_class_labels,average=self.eval_metric_type, zero_division=0)
            class_test_score.append(class_fscore)
            direction_precision,direction_recall,direction_fscore,direction_support=score(test_true_direction_labels,test_pred_direction_labels,average=self.eval_metric_type, zero_division=0)
            direction_test_score.append(direction_fscore)
            total_test_score.append((class_fscore+direction_fscore)/2)
        
        return class_test_score, direction_test_score, total_test_score

    def plot_history(self, history):
        epochs = list(range(1, len(history["total_train_loss"]) + 1))
        figure = plt.figure(figsize=(27, 5))
        figure.suptitle("Learning cross-entropy losses", fontsize=16)
        ax = plt.subplot(131)
        ax.plot(epochs, history["total_train_loss"], label="Train", marker='d', linestyle='-')
        ax.plot(epochs, history["total_dev_loss"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Total ({}) loss".format(self.loss_aggregation_mode))
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        ax = plt.subplot(132)
        ax.plot(epochs, history["class_train_loss"], label="Train", marker='d', linestyle='-')
        ax.plot(epochs, history["class_dev_loss"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Rel. Type loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        ax = plt.subplot(133)
        ax.plot(epochs, history["direction_train_loss"], label="Train", marker='d', linestyle='-')
        ax.plot(epochs, history["direction_dev_loss"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Rel. Direction loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        plt.savefig(self.output_folder+"/"+self.target_task_name+"/learn_loss_curves.png", bbox_inches='tight')
        if self.display_plots:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        figure = plt.figure(figsize=(27, 5))
        figure.suptitle("Validation  {} f1 scores".format(self.eval_metric_type), fontsize=16)
        ax = plt.subplot(131)
        ax.plot(epochs, history["total_dev_score"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Total score")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        ax = plt.subplot(132)
        ax.plot(epochs, history["class_dev_score"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Rel. Type score")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        ax = plt.subplot(133)
        ax.plot(epochs, history["direction_dev_score"], label="Dev", marker='d', linestyle='-')
        ax.set_xticks(epochs)
        ax.set_ylabel("Rel. Direction score")
        ax.set_xlabel("Epoch")
        ax.legend()
        plt.grid(axis='y', linestyle = '--', linewidth = 0.5)
        plt.savefig(self.output_folder+"/"+self.target_task_name+"/dev_score_curves.png", bbox_inches='tight')
        if self.display_plots:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    def fit(self):
        """Learns the pair classifier sub-module on given corpus.

        Returns:
            Nothing. It shows the progress of learning process and the learned tagging sub-module is saved in the base_path.
        """

        criterion = nn.CrossEntropyLoss(reduction="mean")
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.SGD(trainable_parameters, lr=self.initial_learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=self.decay_factor, patience=self.decay_patience, verbose=False)
        
        eval_metric_class = F1Score(num_classes=len(self.corpus.class2idx), average=self.eval_metric_type)
        eval_metric_class = eval_metric_class.to(self.available_device)
        eval_metric_direction = F1Score(num_classes=len(self.corpus.direction2idx), average=self.eval_metric_type)
        eval_metric_direction = eval_metric_direction.to(self.available_device)
        eval_metrics = {"class": eval_metric_class, "direction": eval_metric_direction}
        train_loader, dev_loader, test_loader = self.get_data_loaders()

        logger = self.init_logger()
        
        logger.info("Starting learning")
        logger.info("Checking old output folder, if exists")
        logger.info(self.regenerate_empty_output_directory())
        logger.info("-------")  
        logger.info("DATASET")
        logger.info("-------")
        logger.info("Train {} samples = {} (batch) x {} (sample per batch)".format(len(train_loader)*self.batch_size, len(train_loader), self.batch_size))
        logger.info("Dev   {} samples = {} (batch) x {} (sample per batch)".format(len(dev_loader)*self.batch_size, len(dev_loader), self.batch_size))
        logger.info("Test  {} samples = {} (batch) x {} (sample per batch)".format(len(test_loader)*self.batch_size, len(test_loader), self.batch_size))
        logger.info("-----------")
        logger.info("ANNOTATIONS")
        logger.info("-----------")
        logger.info("Rel. Types: {}".format(self.corpus.class2idx))
        logger.info("Rel. Directions: {}".format(self.corpus.direction2idx))
        logger.info("-----------")
        logger.info("TARGET TASK")
        logger.info("-----------")
        logger.info("{}.".format(self.target_task_name))
        logger.info("---------")
        logger.info("OPTIMIZER")
        logger.info("---------")
        logger.info("Max (train, dev) epochs: {:03d}".format(self.max_epochs))
        logger.info("Initial learning rate: {:.6f}".format(self.initial_learning_rate))
        logger.info("Decay factor: {:.2f}".format(self.decay_factor))
        logger.info("Decay patience period: {:02d} epoch.s".format(self.decay_patience))
        logger.info("Max no-improvement period: {:2d} epoch.s".format(self.max_bad_epochs))
        logger.info("----------")
        logger.info("EVALUATION")
        logger.info("----------")
        logger.info("Evaluation metric: {} avg f1-score".format(self.eval_metric_type))
        logger.info("-------")
        logger.info("LOGGING")
        logger.info("-------")
        logger.info("Logging learning log and checkpoint to: {}\n".format(self.output_folder+"/"+self.target_task_name))
    
        best_dev_performance = -1.0
        nb_bad_epochs = 0
        
        history_train_loss_total = []
        history_train_loss_class = []
        history_train_loss_direction = []

        history_dev_loss_total = []
        history_dev_loss_class = []
        history_dev_loss_direction = []

        history_dev_score_total = []
        history_dev_score_class = []
        history_dev_score_direction = []

        for epoch in range(self.max_epochs):
            
            if nb_bad_epochs >= self.max_bad_epochs:
                logger.info("Epoch {:05d}/{:05d}: exiting training after too many bad epochs.".format(epoch+1, self.max_epochs))
                if self.save_final_model:
                    logger.info("Epoch {:05d}/{:05d}: saving final model before quitting.\n".format(epoch+1, self.max_epochs))
                    torch.save(self.model.state_dict(), self.output_folder+"/"+self.target_task_name+"/final-model.pt")
                break

            else:
                epoch_start_time = time.time()
                #training
                logger.info("Epoch {:05d}/{:05d}: started training with {} train batches:".format(epoch+1, self.max_epochs, len(train_loader)))
                train_class_loss, train_direction_loss, train_total_loss = self.learn_step("train", train_loader, criterion, optimizer, epoch, logger, None)
                history_train_loss_total.append(train_total_loss)
                history_train_loss_class.append(train_class_loss)
                history_train_loss_direction.append(train_direction_loss)
                logger.info("Epoch {:05d}/{:05d}: TRAINIG LOSS (mode = '{}'):".format(epoch+1, self.max_epochs, self.loss_aggregation_mode))
                logger.info("Epoch {:05d}/{:05d}: Total {:.4f}".format(epoch+1, self.max_epochs,train_total_loss))
                logger.info("Epoch {:05d}/{:05d}: Rel. Type: {:.4f}".format(epoch+1, self.max_epochs,train_class_loss))
                logger.info("Epoch {:05d}/{:05d}: Rel. Direction: {:.4f}".format(epoch+1, self.max_epochs,train_direction_loss))
                logger.info("------------------------------------------")
                #validation
                logger.info("Epoch {:05d}/{:05d}: started validation with {} dev batches:".format(epoch+1, self.max_epochs, len(dev_loader)))
                dev_class_loss, dev_direction_loss, dev_total_loss, dev_class_score, dev_direction_score, dev_total_score = self.learn_step("dev", dev_loader, criterion, optimizer, epoch, logger, eval_metrics)
                history_dev_loss_total.append(dev_total_loss)
                history_dev_loss_class.append(dev_class_loss)
                history_dev_loss_direction.append(dev_direction_loss)
                history_dev_score_total.append(dev_total_score)
                history_dev_score_class.append(dev_class_score)
                history_dev_score_direction.append(dev_direction_score)
                logger.info("Epoch {:05d}/{:05d}: VALIDATION SCORE (metric = '{} f1-score'):".format(epoch+1, self.max_epochs, self.eval_metric_type))
                logger.info("Epoch {:05d}/{:05d}: Total {:.4f}".format(epoch+1, self.max_epochs,dev_total_score))
                logger.info("Epoch {:05d}/{:05d}: Rel. Type: {:.4f}".format(epoch+1, self.max_epochs,dev_class_score))
                logger.info("Epoch {:05d}/{:05d}: Rel. Direction: {:.4f}".format(epoch+1, self.max_epochs,dev_direction_score))
                logger.info("------------------------------------------")
                if dev_total_score <= best_dev_performance:
                    nb_bad_epochs += 1
                    logger.info("Epoch {:05d}/{:05d}: no improvement. Number of bad epochs so far: {}".format(epoch+1, self.max_epochs, nb_bad_epochs))
                else:
                    logger.info("Epoch {:05d}/{:05d}: SAVING AS BEST MODEL.".format(epoch+1, self.max_epochs))
                    torch.save(self.model.state_dict(), self.output_folder+"/"+self.target_task_name+"/best-model.pt")
                    with open(self.output_folder+"/"+self.target_task_name+"/model.cfg", "wb") as cfg_file:
                        pickle.dump(self.model, cfg_file)
                    nb_bad_epochs = 0
                    best_dev_performance = dev_total_score
                scheduler.step(dev_total_score)
            
            logger.info("Epoch {:05d}/{:05d}: learning rate: {:.6f}".format(epoch+1, self.max_epochs, optimizer.state_dict()['param_groups'][0]['lr']))
            logger.info("Epoch {:05d}/{:05d}: alpha_class: {:.4f}".format(epoch+1, self.max_epochs, self.model.alpha_class.data.item()))
            logger.info("Epoch {:05d}/{:05d}: alpha_direction: {:.4f}".format(epoch+1, self.max_epochs, self.model.alpha_direction.data.item()))
            logger.info("Epoch {:05d}/{:05d}: duration: {:.4f} secs\n".format(epoch+1, self.max_epochs, time.time() - epoch_start_time))
    
        logger.info("-----------------------")
        logger.info("Testing with {} batches".format(len(test_loader)))
        logger.info("-----------------------")
        class_test_score, direction_test_score, total_test_score = self.test_step(logger, test_loader, self.model)

        
        history = {
            "class_test_score": class_test_score,
            "direction_test_score": direction_test_score,
            "total_test_score": total_test_score,
            "class_train_loss": history_train_loss_class,
            "direction_train_loss": history_train_loss_direction,
            "total_train_loss": history_train_loss_total,
            "class_dev_loss": history_dev_loss_class,
            "direction_dev_loss": history_dev_loss_direction,
            "total_dev_loss": history_dev_loss_total,
            "class_dev_score": history_dev_score_class,
            "direction_dev_score": history_dev_score_direction,
            "total_dev_score": history_dev_score_total,
        }
        if self.log_history_plots:
            self.plot_history(history=history)
        return history