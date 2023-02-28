# Examples

These are simple examples of how to use our python library to fit models.

## SVC

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ml_way.util import load_star_galaxy_dataset


star_galaxy = load_star_galaxy_dataset()

X_train, X_test, y_train, y_test = train_test_split(
  star_galaxy.data,
  star_galaxy.target,
)

svc = SVC()
svc.fit(X_train, y_train)

print(f"Accuracy on training set: {svc.score(X_train, y_train)}")
print(f"Accuracy on test set: {svc.score(X_test, y_test)}")
```

