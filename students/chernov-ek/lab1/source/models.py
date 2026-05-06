import numpy as np
from collections import deque

from evaluators import calculate_gini_impurity, calculate_information_gain


class Node:
    """Contains the information of a node in the Decision Tree."""

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.feature_id = None
        self.threshold = None
        self.is_numeric = False
        self.prediction = None


class DecisionTreeClassifier:
    """Decision Tree Classifier using an ID3-like algorithm with Gini gain."""

    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = list(feature_names)
        self.labels = labels
        self.node = None
        self.gini = calculate_gini_impurity(self.labels)

    def _majority_class(self, labels):
        return max(set(labels), key=list(labels).count)

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        best_feature = None
        best_gain = -1

        for feature_id in feature_ids:
            gain, threshold, is_numeric = calculate_information_gain(
                self.X[x_ids],
                self.labels[x_ids],
                feature_id,
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = (self.feature_names[feature_id], feature_id, threshold, is_numeric)

        return best_feature

    def id3(self):
        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node)
        print('')

    def _id3_recv(self, x_ids, feature_ids, node):
        if not node:
            node = Node()

        labels_in_features = [self.labels[x] for x in x_ids]
        node.prediction = self._majority_class(labels_in_features)

        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node

        if len(feature_ids) == 0:
            node.value = node.prediction
            return node

        best_feature = self._get_feature_max_information_gain(x_ids, feature_ids)
        if best_feature is None:
            node.value = node.prediction
            return node

        best_feature_name, best_feature_id, threshold, is_numeric = best_feature
        node.value = best_feature_name
        node.feature_id = best_feature_id
        node.threshold = threshold
        node.is_numeric = is_numeric
        node.childs = []

        next_feature_ids = [feature_id for feature_id in feature_ids if feature_id != best_feature_id]

        if is_numeric:
            if threshold is None:
                node.value = node.prediction
                node.childs = None
                return node

            numeric_values = self.X[x_ids, best_feature_id].astype(float)
            splits = [
                (f'<= {threshold:.4f}', [x for x, value in zip(x_ids, numeric_values) if value <= threshold]),
                (f'> {threshold:.4f}', [x for x, value in zip(x_ids, numeric_values) if value > threshold]),
            ]
        else:
            feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
            splits = [
                (value, [x for x in x_ids if self.X[x][best_feature_id] == value])
                for value in feature_values
            ]

        for value, child_x_ids in splits:
            child = Node()
            child.value = value
            node.childs.append(child)

            if not child_x_ids:
                child.next = Node()
                child.next.value = node.prediction
                child.next.prediction = node.prediction
            else:
                child.next = self._id3_recv(child_x_ids, next_feature_ids.copy(), child.next)

        return node

    def _predict_one(self, row, node):
        while node and node.childs:
            if node.is_numeric:
                value = float(row[node.feature_id])
                branch_value = node.childs[0].value if value <= node.threshold else node.childs[1].value
            else:
                branch_value = row[node.feature_id]

            next_node = None
            for child in node.childs:
                if child.value == branch_value:
                    next_node = child.next
                    break

            if next_node is None:
                return node.prediction

            node = next_node

        return node.value if node else None

    def predict(self, X):
        if self.node is None:
            raise ValueError('The tree has not been trained. Call id3() first.')

        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return np.array([self._predict_one(row, self.node) for row in X])

    def printTree(self):
        if not self.node:
            return

        nodes = deque()
        nodes.append(self.node)

        while len(nodes) > 0:
            node = nodes.popleft()
            print(node.value)

            if node.childs:
                for child in node.childs:
                    print('({})'.format(child.value))
                    nodes.append(child.next)
            elif node.next:
                print(node.next)
