import numpy as np


def p_r_f(predictions, Y):
    '''
    Returns precision, recall, and F1 score
    '''
    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, y in zip(predictions, Y):
        if pred == y:
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1

    try:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2*p*r / (p + r)
    except:
        # if undefined, just default to all 0
        p, r, f = 0, 0, 0
    return p,r,f


def mean_squared_error(predictions, Y):
    squared_errors = ((pred-y)**2 for pred, y in zip(predictions, Y))
    return sum(squared_errors)/len(Y)


def KFolds(X, Y, num_folds=5):
    # get a list representing indices into X/Y, then shuffle the indices
    indices = np.arange(len(Y))
    np.random.shuffle(indices)

    for i in range(num_folds):
        # For each fold of cross val, grab the next subset of data
        # to be the hold-out portion. Since the indices were pre-shuffled,
        # we can just go in order
        start_index = int(i*len(Y)/num_folds)
        #print(f'start_index = {start_index}')
        end_index = int(start_index + len(Y)/num_folds)
        #print(f'end_index = {end_index}')
        indices_for_holdout = indices[start_index: end_index]
        #print(f'indices_for_holdout = {indices_for_holdout}')
        first_chunk = indices[0:start_index]
        second_chunk = indices[end_index:]
        indices_for_training = np.concatenate((first_chunk, second_chunk))

        training_X, training_Y = X[indices_for_training],  Y[indices_for_training]
        holdout_X, holdout_Y = X[indices_for_holdout],  Y[indices_for_holdout]

        yield training_X, training_Y, holdout_X, holdout_Y

def cross_validation(model, X, Y, task_type, num_folds=5):
    # given a model that can completely transform and predict the given data, X
    # and the expected output, Y, returns a dictionary of evaluation metrics
    # for evaluating num_folds of cross validation
    
    assert(task_type in {'classification', 'regression'})
    predictions = None
    truth = None
    for training_X, training_Y, holdout_X, holdout_Y in KFolds(X, Y, num_folds=num_folds):
        print(f"Fold!")
        model.fit(training_X, training_Y)
        #model.print_tree()
        current_predictions = model.predict(holdout_X)
        #print('preds')
        #print(predictions)
        #print('current preds')
        #print(current_predictions)
        if predictions is None:
            predictions = current_predictions
        else:
            predictions = np.concatenate((predictions, current_predictions))
        if truth is None:
            truth = holdout_Y
        else:
            truth = np.concatenate((truth, holdout_Y))

    #print(f'predictions = {predictions}')
    #print(f'truth = {truth}')

    if task_type == 'classification':
        p,r,f = p_r_f(predictions, truth)
        print(f'p={p}, r={r}, f={f}')
    else:
        mse= mean_squared_error(predictions, truth)
        print(f'mean squared error = {mse}')
