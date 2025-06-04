"""
This implementation is basically the binary version of the MSL.
Unlike PSL, there is no mapping function from scores to probabilities.
Instead, each stage can be interpret as a logistic regression.
All stages share the weights for the features they share.

The feature permutation are provided as an input. it can be fitted using random forrest feature importance or by using the permutations from the greedy search MSL algorithm.
The MIP itself only fits the scores for the features in each stage (and class).

Unlike the discrete LR setting, we do not only have one loss, but a loss for each stage.
The loss is the sum of the losses of all stages.
The loss constraints are therefore also distributed among the stages.
"""

from tqdm import tqdm
from mip import Model, xsum, INTEGER, minimize
import jax.numpy as jnp
import numpy as np
from jax import grad
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optax
from skpsl import MulticlassScoringList
from skpsl.preprocessing import MinEntropyBinarizer
from sklearn.preprocessing import LabelEncoder


def prepare():
    model = Model()

    loss = model.add_var(lb=0, name="loss")
    [
        model.add_var(var_type=INTEGER, lb=-3, ub=3, name=f"w{j},{i}")
        for i in range(d + 1)
        for j in range(c)
    ]

    model.objective = minimize(loss)
    return model


if __name__ == "__main__":
    dataset = fetch_openml(data_id=46764)
    X = dataset.data.values
    y = LabelEncoder().fit_transform(dataset.target.values).astype(int)
    X = MinEntropyBinarizer().fit_transform(X, y)

    X_raw = X.copy()

    n, d = X.shape
    c = len(np.unique(y))  # number of classes

    # Feature permutation (for example, from a random forest feature importance)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importances = rf.feature_importances_
    feature_order = np.argsort(feature_importances)[::-1]  # descending order
    X = X[:, feature_order]  # reorder features based on importance
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))


    # --- JAX loss & gradient ---
    def logloss(w, X, y):
        return sum(
            [
                optax.softmax_cross_entropy_with_integer_labels(
                    X[:, : stage + 1] @ w[:, : stage + 1].T, y
                ).mean()
                for stage in range(d)
            ]
        ) + d * 1e-2 * jnp.mean(w ** 2)


    logloss_grad = grad(logloss)

    model = prepare()
    model.threads = 1
    model.max_mip_gap = 0.1
    model.max_mip_gap_abs = 0.01
    model.max_seconds = 30
    model.verbose = 0

    incumbent = None
    incumbent_loss = float("inf")
    with tqdm() as pbar:
        while True:
            model.optimize()
            w_vars = list(model.vars)[1:]
            w = [var.x for var in w_vars]
            w_np = np.array(w).reshape(c, -1)
            w_jnp = jnp.array(w_np)

            # add cut
            f = float(logloss(w_jnp, X_bias, y))
            g = np.array(logloss_grad(w_jnp, X_bias, y)).flatten().astype(float)
            model += model.vars["loss"] >= f + xsum(
                g[j] * (w_vars[j] - w[j]) for j in range(c * (d + 1))
            )

            # keep incumbent (best real loss!)
            if f < incumbent_loss:
                incumbent = w_np.copy()
                incumbent_loss = f
                incumbent_index = pbar.n + 1

            opt_gap = 1 - model.vars["loss"].x / incumbent_loss
            pbar.update()
            pbar.set_description(
                f"Proxloss: {model.objective_value:.2f}, Loss: {f:.2f}, Gap: {opt_gap:.3f}, Incumbent: {incumbent_index}@{incumbent_loss:.2f}"
            )

            if pbar.n - incumbent_index > 80:
                print("No improvement for 100 iterations, stopping optimization.")
                break

            if opt_gap < 0.02:
                model.max_mip_gap /= 1.5
                # model.max_mip_gap_abs = 0.0001
                continue
            if opt_gap < 1e-2:
                break

    print("Feature order:", feature_order)
    print("Learned integer weights:\n", incumbent.astype(int))
    print("Objective value:", incumbent_loss)
    print()

    msl = MulticlassScoringList(score_set=range(-3, 4), l2=1e-5)
    msl.fit(X, y)
    w = msl.scores[:, [0] + (1 + msl.f_ranks).tolist()]
    X = np.hstack((np.ones((X_raw.shape[0], 1)), X))[
        :, [0] + (1 + msl.f_ranks).tolist()
        ]
    print("Learned weights:\n", w)
    print("Reference performance:", logloss(w, X, y))
    print()

    lr = LogisticRegression(max_iter=100000)
    lr.fit(X, y)
    lr_w = np.hstack((lr.intercept_.reshape(-1, 1), lr.coef_))
    print("Learned float weights:\n", lr_w)
    print("Reference performance:", logloss(lr_w, X_bias, y))
