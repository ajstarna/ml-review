'''
http://www.site.uottawa.ca/~stan/csi5387/boost-tut-ppr.pdf

https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
'''

from ..trees.decision_tree import DecisionTreeStump
#from ..utils.bootstrapping import get_bootstrapped_data_set

from collections import defaultdict
import numpy as np

class AdaBoostClassifier:

    def __init__(self, num_estimators=10):
        self.num_estimators = num_estimators

    def set_index_to_feature_type(self, index_to_feature_type):
        self.index_to_feature_type = index_to_feature_type

    def init_weights(self, X):
        # each training example gets an associated weight,
        # which starts at 1/N
        weights = np.empty(X.shape[0])
        starting_weight = 1/len(weights)
        weights.fill(starting_weight)
        return weights

    def fit(self, X, Y):
        weights = self.init_weights(X) # the weights change on each iteration
        print(weights)
        self.all_estimators = [] # will keep track of each stump and their corresponding weights
        print('Y')
        print(Y)
        for i in range(self.num_estimators):
            print(f'estimator {i} of {self.num_estimators}')
            e_i = DecisionTreeStump()
            e_i.fit(X, Y, index_to_feature_type=self.index_to_feature_type, weights=weights)
            current_preds = e_i.predict(X)
            #print('current preds = ')
            #print(current_preds)
            misclassifications = current_preds != Y
            #print('misclassifications array')
            #print(misclassifications)
            # we get the weights of data that we incorrectly labelled
            misclassification_weights = misclassifications * weights
            #print(misclassification_weights)

            # error is relative to the weights of each data
            error =  np.sum(misclassification_weights) / np.sum(weights)

            # the stage value represents how well the current estimator does at classifying the data
            stage_value = np.log((1 - error) / error)
            
            print(f'stage = {stage_value}')
            # update based on whether we got that datum right or wrong, and how well this estimator did on the whole
            # "This has the effect of not changing the weight if the training instance was classified correctly and 
            # making the weight slightly larger if the weak learner misclassified the instance."
            weights = weights * np.exp(stage_value * misclassifications)

            # then finally normalize the weights so that they sum to 1
            weights = weights / np.sum(weights)
            #print(f'new weights = {weights}')
            
            self.all_estimators.append((e_i, stage_value))
        print("done fitting:")
        print(self.all_estimators)

    def predict_with_mapping(self, e_i, X):
        # we need to map 0 predictions to -1, so that the various estimators can pull against each other
        # the final prediction is based on whether the overall vote is positive or negative
        preds = e_i.predict(X)
        preds[preds == 0] = -1
        return preds

    def predict(self, X):
        all_predictions  = [self.predict_with_mapping(e_i, X) * stage_value for e_i, stage_value in self.all_estimators]
        final = np.zeros(X.shape[0])
        for preds in all_predictions:
            final += preds
        final[final > 0] = 1
        final[final < 0] = 0
        return final
        
        
