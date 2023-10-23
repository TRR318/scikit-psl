import logging
from itertools import combinations, product, repeat
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from skpsl import _ClassifierAtK


from skpsl.helper import create_optimizer
import heapq

class Cascade(BaseEstimator, ClassifierMixin):
    """
    A cascade of probabilistic classifiers.
    A probabilistic classifier that greedily creates a PSL selecting one feature at a time
    """

    def __init__(self, score_set: set, entropy_threshold: float = -1, method="bisect", loss=None):
        """

        :param score_set: Set score values to be considered. Basically feature weights.
        :param entropy_threshold: Shannon Entropy base 2 threshold after which to stop fitting more stages.
        """
        self.loss = self._final_loss
        self.score_set = score_set
        self.entropy_threshold = entropy_threshold
        self.method = method

        self._thresh_optimizer = create_optimizer(method)
        self.logger = logging.getLogger(__name__)
        self.sorted_score_set = np.array(sorted(self.score_set, reverse=True, key=abs))
        self.stage_clfs = []  # type: list[_ClassifierAtK]

   
    def fit(self, X, y, lookahead=1, n_jobs=1):
        """
        This method implements the greedy search algorithm proposed in the IJAR paper (TODO rework citation)

        :param X: _description_
        :param y: _description_
        :param lookahead: _description_, defaults to 1
        :param n_jobs: _description_, defaults to 1
        :param predef_features: _description_, defaults to None
        :param predef_scores: _description_, defaults to None
        """

        number_features = X.shape[1]

        # TODO check how to do binarization
        initial_clf = _ClassifierAtK(features=(), scores=())
        initial_clf.fit(X,y)
        initial_cascade = [initial_clf]
        best_cascades = {initial_cascade}
        best_loss = self.loss(initial_cascade, X, y)    
    
        complexity_heap = []
        heapq.heappush(complexity_heap, (self._complexity(initial_clf, initial_clf)))
        
        while complexity_heap:
            refinement_improving = {}
            refinement_worsening = {}
            # TODO question regarding pseudo_code, does pop_first() pop all elements with the lowest complexity or only one of them? Why can we iterate over it otherwise? Why is there this foreach loop?
            # TODO better variable names
            while complexity_heap:
                _, (H, h) = heapq.heappop(complexity_heap)
                # TODO in pseudo-code the loss is applied to h, but it's defined for entire cascades (like H) 
                h.fit(X,y)

                candidate_cascade = H.append(h)
                cur_loss = self.loss(candidate_cascade, X,y)
                
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_cascades = {candidate_cascade}
                    refinement_worsening |= refinement_improving
                    refinement_improving = {(H,h)}

                elif refinement_improving and cur_loss == best_loss:
                    best_cascades |= candidate_cascade
                    refinement_improving = refinement_improving.add((H,h))
                else:
                    refinement_worsening = refinement_worsening.add((H,h))
            
            for H,h in refinement_improving:
                self._create_nodes(complexity_heap=complexity_heap, cascade=candidate_cascade, stage_clf=h, max_complexity_increase=1)
            
            for H,h in refinement_worsening:
                self._create_nodes(complexity_heap=complexity_heap, cascade=H, stage_clf=h, max_complexity_increase=1)
            

        # retrieve one element from set of best cascades
        best_cascade = next(iter(best_cascades))

        self.stage_clfs = best_cascade

        return self
        
        
    def _create_nodes(self, complexity_heap, cascade, stage_clf, max_complexity_increase=1):
        
        remaining_features = self.features.remove(stage_clf.features)        
        for feature in remaining_features:
            for score in self.score_set:
                features = stage_clf.features + (feature,)
                scores = stage_clf.scores + (score,)
                #TODO check how to do binarization
                clf = _ClassifierAtK(features=features, scores=scores)
                #TODO are the individual cascades and stage_models correct? indices are wild
                h_j = cascade.stage_clfs[-1]
                if self._complexity(clf) - self._complexity(h_j) < max_complexity_increase:
                    heapq.heappush(complexity_heap, (self._complexity(clf)), (cascade, clf))
        pass

    def _final_loss(self, cascade, X, y):
        """
        A simple global loss function for a cascade that evaluates the cascade in terms of the score of its last classifier
        """
        return cascade[-1].score(X,y)
  

    def _complexity(self, stage_clf):
        """_summary_

        :param stage_clf: Classifier to compute complexity of
        :return: Complexity of classifier
        """
        return len(stage_clf.features)

    def predict(self, X, k=-1):
        """
        Predicts a probabilistic scoring list to the given data
        :param X: Dataset to predict the probabilities for
        :param k: Classifier stage to use for prediction
        :return:
        """

        return self.predict_proba(X, k).argmax(axis=1)

    def predict_proba(self, X, k=-1):
        """
        Predicts the probability using the k-th or last classifier
        :param X: Dataset to predict the probabilities for
        :param k: Classifier stage to use for prediction
        :return:
        """
        if not self.stage_clfs:
            raise NotFittedError(
                "Please fit the probabilistic scoring classifier before usage."
            )

        return self.stage_clfs[k].predict_proba(X)

    def score(self, X, y, k=-1, sample_weight=None):
        """
        Calculates the Brier score of the model
        :param X:
        :param y:
        :param k: Classifier stage to use for prediction
        :param sample_weight: ignored
        :return:
        """
        return brier_score_loss(y, self.predict_proba(X, k=k)[:, 1])

    def inspect(self, k=None, feature_names=None) -> pd.DataFrame:
        """
        Returns a dataframe that visualizes the internal model

        :param k: maximum stage to include in the visualization (default: all stages)
        :param feature_names: names of the features.
        :return:
        """
        # TODO 
        pass

    @property
    def features(self):
        return self.stage_clfs[-1].features if self.stage_clfs else []

    @property
    def scores(self):
        return self.stage_clfs[-1].scores if self.stage_clfs else []

    @property
    def thresholds(self):
        return self.stage_clfs[-1].thresholds if self.stage_clfs else []

    def _fit_and_store_clf_at_k(self, X, y, f=None, s=None, t=None):
        f, s, t = f or [], s or [], t or []
        k_clf = _ClassifierAtK(features=f, scores=s, initial_thresholds=t,
                               threshold_optimizer=self._thresh_optimizer).fit(X, y)
        self.stage_clfs.append(k_clf)
        return k_clf.score(X)

    @staticmethod
    def _optimize(
            features, feature_extension, scores, score_extension, thresholds, optimizer, X, y
    ):
        clf = _ClassifierAtK(
            features=features + feature_extension,
            scores=scores + score_extension,
            initial_thresholds=thresholds + [np.nan] * len(feature_extension),
            threshold_optimizer=optimizer
        ).fit(X, y)
        return (
            clf.score(X),
            feature_extension[0],
            score_extension[0],
            clf.thresholds[len(features)],
        )



if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score

    # Generating synthetic data with continuous features and a binary target variable
    X_, y_ = make_classification(random_state=42)

    clf_ = Cascade({-1, 1, 2})
    print("Brier score:", cross_val_score(clf_, X_, y_, cv=5).mean())
