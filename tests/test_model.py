import pytest
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score

# Start Dummy Model Definition (Replace with actual import from your model module)
class SimpleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, param1=1, param2='default'):
        self.param1 = param1
        self.param2 = param2
        self._is_trained = False
        self.classes_ = None
        self._internal_coef = None # Dummy learned attribute

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        # Simulate learning process
        self._internal_coef = np.random.rand(X.shape[1])
        self._is_trained = True
        return self

    def predict(self, X):
        check_is_fitted(self, '_is_trained')
        X = check_array(X)
        if self.classes_ is None or len(self.classes_) == 0:
             raise ValueError("Model classes not set during fit.")

        # Simple deterministic prediction rule for test consistency
        # Predicts based on the first feature relative to 0.5
        # Assumes binary classification [0, 1] for simplicity in this dummy model
        threshold = 0.5
        # Ensure classes_ has at least two elements for this rule, otherwise predict the only class
        if len(self.classes_) >= 2:
            pred_class_1 = self.classes_[1]
            pred_class_0 = self.classes_[0]
            predictions = np.where(X[:, 0] > threshold, pred_class_1, pred_class_0)
        elif len(self.classes_) == 1:
             predictions = np.full(X.shape[0], self.classes_[0])
        else: # Should not happen if fit worked correctly
             raise RuntimeError("Model fit did not set classes_ correctly.")

        return predictions

    def evaluate(self, X, y):
        # predict method already calls check_is_fitted
        X, y = check_X_y(X, y)
        predictions = self.predict(X)
        score = accuracy_score(y, predictions)
        return score

def create_model(param1=1, param2='default'):
    return SimpleModel(param1=param1, param2=param2)

def train_model(model, X, y):
    return model.fit(X, y)

def predict_model(model, X):
    return model.predict(X)

def evaluate_model(model, X, y):
    return model.evaluate(X, y)
# End Dummy Model Definition


# Pytest Fixtures
@pytest.fixture(scope="module")
def dummy_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    # Deterministic labels based on the first feature for consistent testing
    y = (X[:, 0] > 0.5).astype(int)
    return X, y

@pytest.fixture
def untrained_model():
    return create_model()

@pytest.fixture
def trained_model(untrained_model, dummy_data):
    X, y = dummy_data
    return train_model(untrained_model, X, y)

# Test Cases
def test_model_initialization_default():
    model = SimpleModel()
    assert model.param1 == 1
    assert model.param2 == 'default'
    assert not model._is_trained
    assert model.classes_ is None
    assert model._internal_coef is None

def test_model_initialization_custom():
    model = SimpleModel(param1=10, param2='custom')
    assert model.param1 == 10
    assert model.param2 == 'custom'
    assert not model._is_trained

def test_create_model_helper():
    model = create_model(param1=5, param2='test')
    assert isinstance(model, SimpleModel)
    assert model.param1 == 5
    assert model.param2 == 'test'
    assert not model._is_trained

def test_model_training(untrained_model, dummy_data):
    X, y = dummy_data
    model = train_model(untrained_model, X, y)
    assert model._is_trained
    assert hasattr(model, 'classes_')
    assert model.classes_ is not None
    assert len(model.classes_) == 2
    assert np.all(np.isin(model.classes_, [0, 1]))
    assert hasattr(model, '_internal_coef')
    assert model._internal_coef is not None
    assert model._internal_coef.shape == (X.shape[1],)

def test_model_training_returns_self(untrained_model, dummy_data):
     X, y = dummy_data
     model_instance = untrained_model.fit(X, y)
     assert model_instance is untrained_model

def test_model_prediction_shape(trained_model, dummy_data):
    X, _ = dummy_data
    X_test = X[:15]
    predictions = predict_model(trained_model, X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (X_test.shape[0],)

def test_model_prediction_values(trained_model, dummy_data):
    X, _ = dummy_data
    X_test = X[:15]
    predictions = predict_model(trained_model, X_test)
    assert np.all(np.isin(predictions, trained_model.classes_))
    # Check if predictions match the simple deterministic rule
    expected_predictions = (X_test[:, 0] > 0.5).astype(int)
    assert np.array_equal(predictions, expected_predictions)


def test_prediction_before_training(untrained_model, dummy_data):
    X, _ = dummy_data
    with pytest.raises(NotFittedError):
        predict_model(untrained_model, X)

def test_model_evaluation(trained_model, dummy_data):
    X, y = dummy_data
    score = evaluate_model(trained_model, X, y)
    assert isinstance(score, (float, np.floating))
    assert 0.0 <= score <= 1.0
    # Because the dummy prediction logic matches the dummy label generation
    assert score == 1.0

def test_evaluation_before_training(untrained_model, dummy_data):
    X, y = dummy_data
    with pytest.raises(NotFittedError):
        evaluate_model(untrained_model, X, y)

def test_training_with_single_class(untrained_model):
    X = np.random.rand(50, 3)
    y = np.ones(50, dtype=int) # Only class 1
    model = train_model(untrained_model, X, y)
    assert model._is_trained
    assert len(model.classes_) == 1
    assert model.classes_[0] == 1

    X_test = np.random.rand(10, 3)
    predictions = predict_model(model, X_test)
    assert predictions.shape == (10,)
    assert np.all(predictions == 1)

    score = evaluate_model(model, X_test, np.ones(10, dtype=int))
    assert score == 1.0

def test_training_with_empty_data(untrained_model):
    X = np.empty((0, 5))
    y = np.empty((0,))
    with pytest.raises(ValueError, match="0 samples"):
        train_model(untrained_model, X, y)

def test_predict_with_wrong_features(trained_model, dummy_data):
    X_wrong_features = np.random.rand(10, trained_model._internal_coef.shape[0] + 1)
    # check_array in predict should raise ValueError due to inconsistent feature number
    with pytest.raises(ValueError, match="X has [0-9]+ features, but SimpleModel is expecting [0-9]+ features as input"):
         predict_model(trained_model, X_wrong_features)

def test_evaluate_with_wrong_features(trained_model, dummy_data):
    X, y = dummy_data
    X_wrong_features = np.random.rand(y.shape[0], trained_model._internal_coef.shape[0] + 1)
    # check_X_y in evaluate should raise ValueError
    with pytest.raises(ValueError, match="X has [0-9]+ features, but SimpleModel is expecting [0-9]+ features as input"):
         evaluate_model(trained_model, X_wrong_features, y)