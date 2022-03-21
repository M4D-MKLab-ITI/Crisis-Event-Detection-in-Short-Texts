import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score


class Eval:
    """
    Evaluation class for metric calculation and results saving.
    exp_repeats: number of experiments with different seeds [for results averaging]
    n_class: number of output classes for adjusting auc metric
    """
    def __init__(self, exp_repeats, n_class, log_dir):
        # if exp_repeats is not 1 --> activate scoreboard for results averaging
        # scoreboard is not needed for a single seeded experiment
        if exp_repeats != 1:
            self.scoreboard = {'prec': [], 'prec_std': 0, 'rec': [], 'rec_std': 0, 'f1': [], 'f1_std': 0,
                          'acc': [], 'acc_std': 0, 'auc': [], 'auc_std': 0, 'tr_loss': [], 'val_loss': []}
        self.exp_repeats = exp_repeats
        self.n_class = n_class
        self.log_dir = log_dir

    """
    core evaluation function
    1. making prediction
    2. calculating metrics
    3. saving results
    """
    def evaluation(self, x_test, y_test, history, seed, model):
        # prediction phase
        pr = model.predict(x_test)
        pred = np.argmax(pr, axis=1)
        ytrue = np.argmax(y_test, axis=1)

        # calculating metrics
        report = classification_report(ytrue, pred, output_dict=True)
        report["accuracy"] = {
            "accuracy": report["accuracy"]}  # cause only accuracy is not a dict and it crashes in df conversion later
        report["loss"] = {"training": min(history["loss"]), "validation": min(history["val_loss"])}

        # roc auc calculation parameters depend on the loading options
        if self.n_class == 2:
            report["auc_macro"] = {"auc_macro": roc_auc_score(ytrue, pr[:, 1], average='macro')}
            report["auc_weighted"] = {"auc_weighted": roc_auc_score(ytrue, pr[:, 1], average='weighted')}
        else:
            report["auc_macro"] = {"auc_macro": roc_auc_score(ytrue, pr, multi_class="ovo", average='macro')}
            report["auc_weighted"] = {"auc_weighted": roc_auc_score(ytrue, pr, multi_class="ovo", average='weighted')}

        # saving model with best F1-score
        # TODO: 1. make save model a separate function
        # TODO: 2. instead of saving upon condition satisfaction, make it OOP friendly by making properties [best f1, best model]
        if self.exp_repeats != 1:
            if not self.scoreboard["f1"]:
                max_f1 = -1
            else:
                max_f1 = max(self.scoreboard["f1"])
            if max_f1 < report["macro avg"]["f1-score"]:
                model.save('experiments/' + self.log_dir + '/reports/saved_model')
        else:
            model.save('experiments/' + self.log_dir + '/reports/saved_model')

        # saving results
        self.save_results(pred, ytrue, report, str(seed))

        # updating scoreboard
        if self.exp_repeats != 1:
            self.scoreboard["f1"].append(report["macro avg"]["f1-score"])
            self.scoreboard["prec"].append(report["macro avg"]["precision"])
            self.scoreboard["rec"].append(report["macro avg"]["recall"])
            self.scoreboard["acc"].append(report["accuracy"]["accuracy"])
            self.scoreboard["auc"].append(report["auc_macro"]["auc_macro"])
            self.scoreboard["tr_loss"].append(report["loss"]["training"])
            self.scoreboard["val_loss"].append(report["loss"]["validation"])

    """
    save results [both single experiment results and averaging]
    """
    # TODO: call scoreboard result on evaluation func on the last experiment
    def save_results(self, pred=None, ytrue=None, report=None, seed=""):
        prediction_file = 'experiments/' + self.log_dir + '/pred/predictions.csv'
        report_file = 'experiments/' + self.log_dir + '/reports/report.csv'
        if self.exp_repeats != 1:
            prediction_file = prediction_file.replace(".csv", seed + ".csv")
            report_file = report_file.replace(".csv", seed + ".csv")
        if seed == "":
            report = self.scoreboard_report()
        else:
            predictions = dict()
            for i, value in enumerate(pred):
                index = i
                predictions[index] = {"prediction": value,
                                      "label": ytrue[i]}
            df = pd.DataFrame.from_dict(predictions, orient="index")
            df.to_csv(prediction_file)

        # saving the report
        df = pd.DataFrame.from_dict(report, orient="index")
        df.to_csv(report_file)
        print(report)

    """
    scoreboard_report makes a report of the average values and stddevs of all reports
    scoreboard is defined in the global scope of the program.
    it is lists of all metrics of different experiments performed.
    """
    def scoreboard_report(self):
        # iterating through the scoreboard
        for key, value in self.scoreboard.copy().items():
            # condition that gives the lists of metrics to get the averages and the stds
            if "std" not in key:
                # if there is an std section in the scoreboard get std for current metric
                try:
                    if self.scoreboard[key + "_std"] == 0:
                        self.scoreboard[key + "_std"] = np.std(np.asarray(value))
                except KeyError:
                    pass
                self.scoreboard[key] = np.mean(np.asarray(value))  # replace list with the mean of the list
        return self.scoreboard