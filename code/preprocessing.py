import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np


PATH = Path.cwd()
DATA_PATH = PATH / "data"

os.makedirs(DATA_PATH / "processed", exist_ok=True)

df = pd.read_csv(DATA_PATH / "raw" / "water_potability.csv")
X, y = df.drop(columns=["Potability"]), df["Potability"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

imputer = SimpleImputer(strategy="mean")

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    np.asarray(imputer.transform(X_test)),
    columns=X_test.columns,
    index=X_test.index
)

outlier_mask = pd.Series(False, index=X_train.index)

for col in X_train_imputed.columns:
    q1 = X_train_imputed[col].quantile(0.25)
    q3 = X_train_imputed[col].quantile(0.75)

    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr

    outlier_mask |= ~X_train_imputed[col].between(lower, upper)

X_train_clean = X_train_imputed.loc[~outlier_mask]
y_train_clean = y_train.loc[~outlier_mask]

train = pd.concat([X_train_clean, y_train_clean], axis=1)
test = pd.concat([X_test_imputed, y_test], axis=1)

train.to_csv(DATA_PATH / "processed" / "train.csv")
test.to_csv(DATA_PATH / "processed" / "test.csv")
