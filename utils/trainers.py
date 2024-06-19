# device = torch.device("cpu")
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch


def svm_normalize_train_test(train_set, test_set=None, standardize_obj=StandardScaler):
    result = []
    scaler = standardize_obj()
    new_train_set = scaler.fit_transform(train_set)
    if test_set is not None:
      new_test_set = scaler.transform(test_set)
      return new_train_set, new_test_set
    else:
      return new_train_set


def svm_normalize_embedded_dl(embedded_dl):
    X = []
    for i, (outputs_dict, tdict) in enumerate(embedded_dl):
      outputs = outputs_dict['outputs']
      outputs = outputs.detach().cpu().numpy()
      for k, frame in enumerate(outputs):
        X.append(frame)
    
    normed_X = svm_normalize_train_test(X)
    new_embedded_dl = []
    tracker = 0
    for i in range(len(embedded_dl)):
      odict, tdict = embedded_dl[i]
      size = odict['outputs'].shape[0]
      odict['outputs'] = torch.from_numpy(normed_X[tracker:tracker + size])
      new_embedded_dl.append((odict, tdict))
      tracker = tracker + size
    return new_embedded_dl


def train_linear_regressor(X, y, test_size=0.2, random_state=None, normalize=False):
    if normalize:
      X = svm_normalize_train_test(X)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X)

    # Calculate mean squared error on the test set
    r2 = r2_score(y, y_pred)

    return model, r2


def train_and_evaluate_svm(X_train, y_train, X_test, y_test, normalize=False):
    if normalize:
      X_train, X_test = svm_normalize_train_test(X_train, X_test)
    if len(set(y_train)) == 1:
       print('bad luck! only one class represented. Try again and check the class distribution.')
       return None, None

    # Initialize the SVM classifier
    clf = svm.SVC(decision_function_shape='ovo', tol=1e-3)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy