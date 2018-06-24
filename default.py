import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
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