import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from util import load_star_galaxy_dataset


def display_scores(model):
    predictions = model.predict(X_test)
    print(
        classification_report(
            y_true=y_test,
            y_pred=predictions,
            target_names=star_galaxy.target_names,
        )
    )
    C = confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        normalize="true",
    )
    print(f"true negatives: {C[0,0]}")
    print(f"false negatives: {C[1,0]}")
    print(f"true positives: {C[1,1]}")
    print(f"false positives: {C[0,1]}")

star_galaxy = load_star_galaxy_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    star_galaxy.data,
    star_galaxy.target,
    test_size=0.3,
)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)

predictions = knn_model.predict(X_test)
display_scores(model=knn_model)




