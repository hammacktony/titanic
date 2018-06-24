# A host of Scikit-learn models
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, r2_score, roc_auc_score, classification_report, accuracy_score, roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.base import clone
from sklearn.model_selection import KFold
from mlens.ensemble import SuperLearner


global SEED
SEED = 0

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, kernel='rbf', probability=True)
    knn = KNeighborsClassifier(n_neighbors=5)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, criterion = 'entropy', max_features=3, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              # 'naive bayes': nb, Removed due to poor performance in ensemble and GBM
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models


def train_predict(model_list,xtrain, xtest, ytrain, ytest):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    scores = dict()
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
        scores[m]=score
    print("Done.\n")
    return scores

def prepare_data(train,test):

    train['Sex'] = pd.get_dummies(train['Sex']) #0 for male, one for female
    # Create Dummy Variable for port where they embarked from
    train = pd.get_dummies(train, columns=['Embarked'])
    train = train.drop(['Name','Ticket','Cabin'], axis=1)

    # Cleaning Test Data -----------------
    # Create Dummy Variable for Sex
    test['Sex'] = pd.get_dummies(test['Sex']) #0 for male, one for female
    # Create Dummy Variable for port where they embarked from
    test = pd.get_dummies(test, columns=['Embarked'])
    # Drop Unneeded Columns
    test = test.drop(['Name','Ticket','Cabin'], axis=1)

    #Filled in NaN in age with mean age
    train['Age'] = train['Age'].fillna(train['Age'].mean())
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

    # Getting Data
    X = train.iloc[:, 2:].values
    y = train.iloc[:, 1].values
    # Scaling data 
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
     
    return scaled_X,y,test 

# def find_lowest_auc(scores,P):
#     key_min = min(scores.keys(), key=(lambda k: scores[k])) 
#     print('The algorithm with the lowest auc score is '+key_min+' with a score of %.3f.\n' % scores[key_min])   
#     # include = [c for c in P.columns if c not in [key_min]]
#     # print("Truncated ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.loc[:, include].mean(axis=1)))

# def plot_correlation(P):
#     from mlens.visualization import corrmat
#     corrmat(P.corr(), inflate=False)
#     plt.show()

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])
        
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()

def train_base_learners(base_learners, inp, out, verbose=True):
        """Train all base learners in the library."""
        if verbose: print("\nFitting models.")
        for i, (name, m) in enumerate(base_learners.items()):
            if verbose: print("%s..." % name, end=" ", flush=False)
            m.fit(inp, out)
            if verbose: print("done")

def predict_base_learners(pred_base_learners, inp, verbose=True):
    """Generate a prediction matrix."""
    P = np.zeros((inp.shape[0], len(pred_base_learners)))

    if verbose: print("\nGenerating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict_proba(inp)
        # With two classes, need only predictions for one class
        P[:, i] = p[:, 1]
        if verbose: print("done")

    return P

def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """Generate predictions from the ensemble."""
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    return P_pred, meta_learner.predict_proba(P_pred)[:, 1]


def stacking(base_learners, meta_learner, X, y, generator):
    """Simple training routine for stacking."""

    # Train final base learners for test time
    print("\nFitting final base learners...", end="")
    train_base_learners(base_learners, X, y, verbose=False)
    print("done")

    # Generate predictions for training meta learners
    # Outer loop:
    print("\nGenerating cross-validated predictions...")
    cv_preds, cv_y = [], []
    for i, (train_idx, test_idx) in enumerate(generator.split(X)):

        fold_xtrain, fold_ytrain = X[train_idx, :], y[train_idx]
        fold_xtest, fold_ytest = X[test_idx, :], y[test_idx]

        # Inner loop: step 4 and 5
        fold_base_learners = {name: clone(model)
                              for name, model in base_learners.items()}
        train_base_learners(
            fold_base_learners, fold_xtrain, fold_ytrain, verbose=False)

        fold_P_base = predict_base_learners(
            fold_base_learners, fold_xtest, verbose=False)

        cv_preds.append(fold_P_base)
        cv_y.append(fold_ytest)
        print("Fold %i done" % (i + 1))

    print("CV-predictions done")
    
    # Be careful to get rows in the right order
    cv_preds = np.vstack(cv_preds)
    cv_y = np.hstack(cv_y)

    # Train meta learner
    print("Fitting meta learner...", end="")
    meta_learner.fit(cv_preds, cv_y)
    print("done")

    return base_learners, meta_learner



if __name__ =='__main__':

    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')

    models = get_models()

    X, y, test = prepare_data(train,test) 
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 0)
    P = train_predict(models, xtrain, xtest, ytrain, ytest)
    scores = score_models(P, ytest)
    find_lowest_auc(scores,P)
    # plot_correlation(P)
    print("Ensemble (Averaging) ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))
    plot_roc_curve(ytest, P.values, P.mean(axis=1), list(P.columns), "ensemble")
   
    # Meta Learner
    base_learners = get_models()
    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        max_features=4,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.005, 
        random_state=SEED
    )
    xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(
    xtrain, ytrain, test_size=0.5, random_state=SEED)

    train_base_learners(base_learners, xtrain_base, ytrain_base)

    P_base = predict_base_learners(base_learners, xpred_base)
    meta_learner.fit(P_base, ypred_base)
    P_pred, p = ensemble_predict(base_learners, meta_learner, xtest)
    print("\nEnsemble (GBM) ROC-AUC score: %.3f" % roc_auc_score(ytest, p))

    # Train with stacking
    cv_base_learners, cv_meta_learner = stacking(
        get_models(), clone(meta_learner), xtrain, ytrain, KFold(2))

    P_pred, p = ensemble_predict(cv_base_learners, cv_meta_learner, xtest, verbose=False)
    print("\nEnsemble (Stacking) ROC-AUC score: %.3f" % roc_auc_score(ytest, p))


    # Instantiate the ensemble with 10 folds
    ensemble = SuperLearner(
        folds=10,
        random_state=SEED,
        verbose=2,
        backend="multiprocessing"
    )

    # Add the base learners and the meta learner
    ensemble.add(list(base_learners.values()), proba=True) 
    ensemble.add_meta(meta_learner, proba=True)

    # Train the ensemble
    ensemble.fit(xtrain, ytrain)

    # Predict the test set
    p_sl = ensemble.predict_proba(xtest)

    print("\nSuper Learner ROC-AUC score: %.3f" % roc_auc_score(ytest, p_sl[:, 1]))
    plot_roc_curve(ytest, p.reshape(-1, 1), P.mean(axis=1), ["Simple average"], "Super Learner")

    print('-------------------------------------')
    print(test.head())
    y_pred = ensemble.predict(test.iloc[:,1:].values)
    print(y_pred)
