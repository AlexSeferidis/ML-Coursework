import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def get_entropy(dataset):
    """ Calculate the entropy of a dataset.

    Args:
        dataset (np.ndarray): The dataset for which to calculate entropy. Has shape of (N, K)

    Returns:
        entropy (float): The calculated entropy value.
    """ 

    total = dataset.shape[0] # total no .of samples

    unique_label, unique_count = np.unique(dataset, return_counts=True)
    prob = unique_count / total
    
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def get_infogain(dataset, left, right):
    """ Calculate the information gain from a split.

    Args:
        dataset (np.ndarray): The original dataset before the split. Has shape (N, K)
        left (np.ndarray): The left subset after the split. Has shape (N, K)
        right (np.ndarray): The right subset after the split. Has shape (N, K)

    Returns:
        info_gain (float): The information gain from the split.
    """ 

    dataset_total = dataset.shape[0]
    left_total = left.shape[0]
    right_total = right.shape[0]
    
    dataset_entropy = get_entropy(dataset)
    left_entropy = get_entropy(left)
    right_entropy = get_entropy(right)
    
    info_gain =  dataset_entropy - ((left_total/dataset_total * left_entropy) + (right_total/dataset_total * right_entropy))
    
    return info_gain


def find_split(dataset):
    """ Find the best split for the training dataset.

    Args:
        dataset (np.ndarray): The dataset to find the best split for. Has shape (N, K)

    Returns:
        bestSplit (tuple): A tuple containing the best attribute index, split value, left subset, and right subset.
    """  

    max_gain = - float('inf')
    for i in range(dataset.shape[1]-1):
        sorted_indexes = np.argsort(dataset[:, i])
        sorted_dataset = dataset[sorted_indexes]

        for j in range(sorted_dataset.shape[0]-1):
            curr_val = sorted_dataset[j][i]
            next_val = sorted_dataset[j+1][i]
            
            if curr_val != next_val:
                split_val = (curr_val + next_val)/2

                left_set = sorted_dataset[sorted_dataset[:, i] < split_val]
                right_set = sorted_dataset[sorted_dataset[:, i] >= split_val]

                info_gain = get_infogain(sorted_dataset, left_set, right_set)

                if info_gain > max_gain:
                    max_gain = info_gain
                    bestSplit = (i, split_val, left_set, right_set)
    return bestSplit


def decision_tree_learning(training_dataset, depth):
    """ Recursively builds a decision tree by defining each node as a dictionary.

    Args:
        training_dataset (np.ndarray): The dataset to build the tree from. Has shape (N, K)
        depth (int): The current depth of the tree.

    Returns:
        (node, depth): A tuple containing the tree node and the depth of the tree.
    """    

    isleaf = len(np.unique(training_dataset[:, -1])) == 1
    
    if isleaf:
        leaf = {
            "attribute": "Leaf",
            "value": training_dataset[0, -1],
            "left": None,
            "right": None,
            "leafnode": isleaf
        }
        return (leaf , depth)
    
    else:
        split = find_split(training_dataset)
        
        node = {
            "attribute": split[0], 
            "value": split[1], 
            "left": None , 
            "right": None, 
            "leafnode": isleaf 
        }

        left_branch, l_depth = decision_tree_learning(split[2], depth+1) 
        right_branch, r_depth = decision_tree_learning(split[3], depth+1)
        
        node["left"] = left_branch
        node["right"] = right_branch
        return(node, max(l_depth, r_depth))


def predict(tree, test_dataset):
    """ Generate predictions for a test dataset using the decision tree.

    Args:
        tree (dict): The decision tree to use for predictions.
        test_dataset (np.ndarray): The dataset to predict. Has shape (N, K)

    Returns:
        predictions (np.ndarray): An array of predicted class labels. Has shape (N, )
    """   

    predictions = np.zeros(test_dataset.shape[0])
    
    for i in range(test_dataset.shape[0]):
        predictions[i] = predict_helper(tree, test_dataset[i])
    
    return predictions

def predict_helper(tree, test_data):
    """ Helper function to traverse the decision tree for prediction.

    Args:
        tree (dict): The current node of the decision tree.
        test_data (np.ndarray): The instance to predict. Has shape (N, )

    Returns:
        value (float): The predicted class label.
    """  

    if tree["leafnode"]:
        return tree["value"]
    
    attribute = tree["attribute"]
    value = tree["value"]

    if test_data[attribute] < value:
        return predict_helper(tree["left"], test_data)
    else:
        return predict_helper(tree["right"], test_data)


def conf_matrix_gen(confusion, predictions, gold_labels):
    """ Generate a confusion matrix based on the predictions and the golden labels.

    Args:
        confusion (np.ndarray): The confusion matrix to update. Has shape (N, N)
        predictions (np.ndarray): The predicted class labels. Has shape (N, )
        gold_labels (np.ndarray): The golden class labels. Has shape (N, )

    Returns:
        confusion (np.ndarray): The updated confusion matrix. Has shape (N, N)
    """

    for i in range(gold_labels.shape[0]):
        confusion[int(gold_labels[i])-1, int(predictions[i])-1] += 1

    return confusion


def calculate_accuracy(conf_matrix):
    """ Calculate the accuracy from the confusion matrix.
    Accuracy = correct predictions / total predictions

    Args:
        conf_matrix (np.ndarray): The confusion matrix.

    Returns:
        accuracy (float): The calculated accuracy.
    """

    correct_predictions = np.trace(conf_matrix)
    total_predictions = np.sum(conf_matrix)
    
    accuracy = correct_predictions / total_predictions
    return accuracy


def calculate_metrics(conf_matrix):
    """ Calculate precision, recall, and f1 score from the confusion matrix.
        Precision = Tp / (Tp + Fp)
        Recall = Tp / (Tp + Fn)
        F1 = 2 * (Precision * Recall) / (Precision + Recall)


    Args:
        conf_matrix (np.ndarray): The confusion matrix. Has shape (N, N)

    Returns:
        precision (np.ndarray): Array of precision values for each class. Has shape (N, )
        recall (np.ndarray): Array of recall values for each class. Has shape (N, )
        f1 (np.ndarray): Array of F1 scores for each class. Has shape (N, )
    """  

    precision = np.zeros(conf_matrix.shape[0])
    recall = np.zeros(conf_matrix.shape[0])
    f1 = np.zeros(conf_matrix.shape[0])
    
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    return precision, recall, f1


def split(splits, instances, random_generator=default_rng()):
    """ Split the dataset into random subsets.

    Args:
        splits (int): The number of splits to create.
        instances (int): The total number of instances in the dataset.
        random_generator (np.random.Generator): Random number generator for shuffling.

    Returns:
        split_indices (np.ndarray): An array of the shuffled indices split into the number of splits. Has shape (N, K)
    """  

    shuffled_indices = random_generator.permutation(instances)
    split_indices = np.array_split(shuffled_indices, splits)

    return np.array(split_indices)


def cv_evaluate(dataset, conf_mat, num_folds=10):
    """ Evaluate the model using cross-validation.

    Args:
        dataset (np.ndarray): The given dataset. Has shape (N, K)
        conf_mat (np.ndarray): The confusion matrix to update. (N, N)
        num_folds (int): The number of folds for cross-validation.

    Returns:
        metrics (tuple): Calculated accuracy, precision, recall, and F1 scores.
    """    
    
    split_indices = split(num_folds, dataset.shape[0])
    print(split_indices.shape)
    Perfs = []
    for i in range(split_indices.shape[0]):
        test_data = dataset[split_indices[i]]
        train_indices = np.ndarray.flatten(np.concatenate((split_indices[:i], split_indices[i+1:])))
        train_data = dataset[train_indices]
        
        tree, _ = decision_tree_learning(train_data, 0)
        predictions = predict(tree, test_data)
        gold_labels = test_data[:, -1]
        conf_mat = conf_matrix_gen(conf_mat, predictions, gold_labels)
        print("Loading")
    
    metrics = evaluate(conf_mat)
    print(conf_mat)
    print("Accuracy", float(metrics[0]))
    print("Precision", metrics[1])
    print("Recall", metrics[2])
    print("F1 measures", metrics[3])

    return metrics


def evaluate(conf_mat):
    """ Use the confusion matrix to calculate accuracy, precision, recall, and F1 score using the previous helper functions.

    Args:
        conf_mat (np.ndarray): The confusion matrix. Has shape (N, N)

    Returns:
        tuple: Accuracy, precision, recall, and F1 scores.
    """  
    
    accuracy = calculate_accuracy(conf_mat)
    precision, recall, f1 = calculate_metrics(conf_mat)
    return accuracy, precision, recall, f1

def get_tree_depth(node):
    """ Helper function to calculate maximum depth of the tree
    
    Args:
        node (dict): Reference node to calculate depth from
    
    Returns:
        max_depth (float): Value for maximum depth of tree
    """
    if node is None or node["leafnode"]:
        return 0
    left_depth = get_tree_depth(node["left"])
    right_depth = get_tree_depth(node["right"])
    return max(left_depth, right_depth) + 1


def plot_tree(node, depth=0, x=0, y=0, x_offset=10, y_offset=2, ax=None, max_depth=None):
    """ Plot the decision tree.

    Args:
        node (dict): The current node of the decision tree to plot.
        depth (int): The current depth of the tree (default is 0).
        x (float): The x-coordinate for the current node's position (default is 0).
        y (float): The y-coordinate for the current node's position (default is 0).
        x_offset (float): The horizontal offset between nodes (default is 2).
        y_offset (float): The vertical offset between levels (default is 1).
        ax (matplotlib.axes.Axes): The axes on which to plot the tree (default is None, which creates a new figure).
        max_depth (float): Value for maximum depth of tree, used to dynamically adjust spacing

    """  
        
    if ax is None:
        if max_depth is None:
            max_depth = get_tree_depth(node)
        fig_width = max(12, max_depth * 4)
        fig_height = max(10, max_depth * 2.5)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')  # Turn off the axis

    if node["leafnode"]:
        ax.text(x, y, f"Leaf: {node['value']}", ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))
        return
    
    ax.text(x, y, f"{node['attribute']} < {node['value']:.2f}", ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="blue", facecolor="white"))
    
    # dynamic adjustment of horizontal spacing based (try to prevent overlapping)
    left_x = x - x_offset / (2.1 ** depth)
    right_x = x + x_offset / (2.1  ** depth)
    child_y = y - y_offset
    # left_x = x - x_offset / (depth + 1) * (max_depth - depth + 1)
    # right_x = x + x_offset / (depth + 1) * (max_depth - depth + 1)
    # child_y = y - y_offset # constant vertical distance

    ax.plot([x, left_x], [y, child_y], color='black')
    ax.plot([x, right_x], [y, child_y], color='black')
    
    plot_tree(node["left"], depth + 1, left_x, child_y, x_offset, y_offset, ax, max_depth=max_depth)
    plot_tree(node["right"], depth + 1, right_x, child_y, x_offset, y_offset, ax, max_depth=max_depth)
    
    if depth == 0:
        plt.savefig("tree_plot.png", format='png', bbox_inches='tight')
        plt.close(fig)


def gen_tree(dataset):
    """ Generate a decision tree and evaluate its performance.

    Args:
        dataset (np.ndarray): The dataset to generate the tree from. Has shape (N, K)
    """   

    class_labels = np.unique(dataset[:,-1])
    conf_mat = np.zeros((len(class_labels), len(class_labels)), dtype=np.int64)
    cv_evaluate(dataset, conf_mat)
    return 0


# noisy = np.loadtxt('wifi_db/noisy_dataset.txt')
clean = np.loadtxt('wifi_db/clean_dataset.txt')

gen_tree(clean)
tree, _ = decision_tree_learning(clean, 0)
plot_tree(tree)

