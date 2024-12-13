# Decision Tree Classifier

**NOTE: TREE VISUALISATION IMAGES CAN BE SEEN IN FULL RESOLUTION IN ['tree_images'](tree_images) FOLDER**

This repository contains an implementation of a Decision Tree Classifier using Python and NumPy. The classifier is designed to handle datasets for classification tasks, and it includes functionality for calculating metrics such as accuracy, precision, recall, and F1 score. The implementation also supports cross-validation for model evaluation.

## Features

- **Entropy Calculation**: Computes the entropy of a dataset to measure the impurity.
- **Information Gain**: Calculates the information gain from potential splits in the dataset.
- **Decision Tree Learning**: Recursively builds a decision tree based on the training dataset.
- **Prediction**: Generates predictions for new instances using the trained decision tree.
- **Confusion Matrix Generation**: Creates a confusion matrix to evaluate the performance of the classifier.
- **Cross-Validation**: Implements k-fold cross-validation to assess the model's performance on different subsets of the data.
- **Performance Metrics**: Calculates accuracy, precision, recall, and F1 score.

## Requirements

- Python 3.x
- NumPy
- Matplotlib (optional, for visualization)

## Usage

1. **Load Your Data**: Prepare your dataset in a suitable format (e.g., CSV or TXT) and ensure it is structured with features in columns and the target label in the last column.

2. **Run the Code**: Use the provided functions to load your dataset and generate a decision tree. The code includes examples for loading noisy and clean datasets.

3. **Evaluate the Model**: The model will automatically evaluate its performance using cross-validation and print the confusion matrix along with accuracy, precision, recall, and F1 scores.

### Example

```python
# Load the data from the .txt file
noisy = np.loadtxt('wifi_db/noisy_dataset.txt')
clean = np.loadtxt('wifi_db/clean_dataset.txt')

# Generate and evaluate decision trees
gen_tree(clean)
gen_tree(noisy)
```
