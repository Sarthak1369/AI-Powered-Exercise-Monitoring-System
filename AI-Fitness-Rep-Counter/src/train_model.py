import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('coords.csv')

# =========================
# FEATURES & LABELS
# =========================

X = df.drop('class', axis=1)
y = df['class']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=1234
)

# =========================
# PIPELINES
# =========================

pipelines = {

    'lr': make_pipeline(
        StandardScaler(),
        LogisticRegression()
    ),

    'rc': make_pipeline(
        StandardScaler(),
        RidgeClassifier()
    ),

    'rf': make_pipeline(
        StandardScaler(),
        RandomForestClassifier()
    ),

    'gb': make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier()
    )
}

# =========================
# TRAIN MODELS
# =========================

fit_models = {}

for algo, pipeline in pipelines.items():

    model = pipeline.fit(X_train, y_train)

    fit_models[algo] = model

# =========================
# EVALUATION
# =========================

for algo, model in fit_models.items():

    yhat = model.predict(X_test)

    print("\n===================")
    print(algo)
    print("===================")

    print(
        "Accuracy:",
        accuracy_score(y_test, yhat)
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            yhat,
            average="binary"
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            yhat,
            average="binary"
        )
    )

# =========================
# SAVE RANDOM FOREST MODEL
# =========================

with open('pushup.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

print("\nModel Saved Successfully")