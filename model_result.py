from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score


class ValidationResult:
    def __init__(
            self,
            model,
            roc_auc,
            current_accuracy_score,
            current_precision_score,
            current_recall_score,
            current_confusion_matrix,
            report,
            ):
        self.model = model
        self.report = report
        self.recall_score = current_recall_score
        self.precision_score = current_precision_score
        self.accuracy_score = current_accuracy_score
        self.roc_auc = roc_auc
        self.confusion_matrix = current_confusion_matrix

    def __str__(self):
        return '\n'.join([
            f'model: {self.model.name}',
            f'accuracy_score: {self.accuracy_score}',
            f'precision_score: {self.precision_score}',
            f'recall_score: {self.recall_score}',
            f'roc_auc_scope: {self.roc_auc}',
            f'confusion_matrix:\n{self.confusion_matrix}',
            f'report:\n{self.report}',
        ])


class ModelWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def validate(self, x_test, y_test):
        predict = self.model.predict(x_test)
        proba = self.model.predict_proba(x_test)

        return ValidationResult(
            self,
            roc_auc_score(y_test, predict),
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            confusion_matrix(y_test, predict),
            classification_report(y_test, predict),
        )
