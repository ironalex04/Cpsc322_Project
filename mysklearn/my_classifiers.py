from mysklearn import myutils
import math

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        # global majority class label (used as a safe fallback)
        self._default_label = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # store training data
        self.X_train = X_train
        self.y_train = y_train
    
        # compute global majority label for fallback predictions
        self._default_label = self._majority_vote(y_train)
    
        if not X_train:
            # edge case — empty dataset
            self.tree = None
            return
    
        # Build TDIDT instances: attributes + class label as last element
        instances = [x + [y] for x, y in zip(X_train, y_train)]
        num_attributes = len(X_train[0])
        available_attributes = list(range(num_attributes))
    
        # Recursively build the tree
        self.tree = self._tdidt(instances, available_attributes, parent_count=None)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.tree is None:
            # No model; return the global majority label for all instances
            return [self._default_label for _ in X_test]

        y_pred = []
        for instance in X_test:
            y_pred.append(self._predict_instance(instance, self.tree))
        return y_pred

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if self.tree is None:
            print("<empty tree>")
            return

        # Start DFS from root with no conditions
        self._print_rules_recursive(self.tree, [], attribute_names, class_name)


    def _majority_vote(self, labels):
        """Computes the majority class label from a list of labels.

        Args:
            labels(list of obj): A list of target class labels.

        Returns:
            obj: The label that appears most frequently in labels. If there is
                a tie for the highest frequency, the alphabetically first label
                is returned.

        Notes:
            This helper is used when TDIDT needs to create a leaf node in cases
            where there is no clear attribute to split on or when a branch has
            no training instances.
        """
        if not labels:
            return None
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        max_count = max(counts.values())
        # tie-break: choose alphabetically smallest label
        best_labels = [lab for lab, c in counts.items() if c == max_count]
        best_labels.sort()
        return best_labels[0]

    def _compute_entropy(self, labels):
        """Computes the entropy of a set of class labels.

        Args:
            labels(list of obj): A list of target class labels.

        Returns:
            float: The entropy value (in bits) of the label distribution. If the
                list is empty, 0.0 is returned.

        Notes:
            Entropy is defined as sum over classes of -p(c) * log2(p(c)), where
            p(c) is the proportion of labels in that class. This helper is used
            by the attribute selection method to compute information gain.
        """
        n = len(labels)
        if n == 0:
            return 0.0
        counts = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        entropy = 0.0
        for c in counts.values():
            p = c / n
            entropy -= p * math.log2(p)
        return entropy

    def _partition_instances(self, instances, att_index):
        """Partitions instances by the value of a given attribute index.

        Args:
            instances(list of list of obj): The training instances. Each inner
                list represents one instance where the last element is the
                class label.
            att_index(int): The index of the attribute to partition on.

        Returns:
            dict: A dictionary mapping attribute values (obj) to a list of
                instances (list of list of obj) that have that value.

        Notes:
            This helper is used by the TDIDT algorithm to create branches for
            each value of a chosen attribute.
        """
        partitions = {}
        for row in instances:
            val = row[att_index]
            partitions.setdefault(val, []).append(row)
        return partitions

    def _select_attribute(self, instances, available_attributes):
        """Selects the best attribute to split on using information gain.

        Args:
            instances(list of list of obj): The training instances at the
                current node. The last element of each instance is the class
                label.
            available_attributes(list of int): The indices of attributes that
                are still available for splitting.

        Returns:
            int: The index of the selected attribute that maximizes information
                gain. If there is a tie in information gain between multiple
                attributes, the attribute with the smallest index is returned.

        Notes:
            Information gain is computed as:
                gain(S, A) = entropy(S) - sum_v (|S_v| / |S|) * entropy(S_v)
            where S_v is the subset of instances with attribute A equal to
            value v.
        """
        # entropy at the current node
        labels = [row[-1] for row in instances]
        base_entropy = self._compute_entropy(labels)
        best_gain = -1.0
        best_att = None

        n = len(instances)
        for att_index in available_attributes:
            partitions = self._partition_instances(instances, att_index)
            # weighted entropy of partitions
            new_entropy = 0.0
            for subset in partitions.values():
                weight = len(subset) / n
                subset_labels = [row[-1] for row in subset]
                new_entropy += weight * self._compute_entropy(subset_labels)
            info_gain = base_entropy - new_entropy
            if (info_gain > best_gain or
                (math.isclose(info_gain, best_gain) and
                 (best_att is None or att_index < best_att))):
                best_gain = info_gain
                best_att = att_index

        # fallback (should not occur if available_attributes is non-empty)
        if best_att is None:
            best_att = available_attributes[0]
        return best_att

    def _tdidt(self, instances, available_attributes, parent_count=None):
        """Recursively builds a decision tree using the TDIDT algorithm.

        Args:
            instances(list of list of obj): The training instances that reach
                the current node. Each inner list represents an instance where
                the last element is the class label.
            available_attributes(list of int): The attribute indices that are
                still available for splitting at this node.
            parent_count(int or None): The number of instances that reached the
                parent of this node. This is used to populate the "total"
                value stored in each leaf node. If None, the current number of
                instances is used.

        Returns:
            list: A nested list representation of the decision tree rooted at
                this node. Internal nodes have the form:
                    ["Attribute", "att#", ["Value", v1, subtree1], ...]
                Leaf nodes have the form:
                    ["Leaf", label, count_label, total_at_parent]

        Notes:
            Base cases:
                - All instances at this node have the same class label.
                - No attributes remain to split on (majority vote is used).
            Recursive case:
                - Select the best attribute using information gain.
                - Partition instances by attribute values.
                - Recursively build subtrees for each partition.
        """
        # class labels at this node
        labels = [row[-1] for row in instances]

        # base case 1: all instances have the same class
        if all(lab == labels[0] for lab in labels):
            label = labels[0]
            count_label = len(labels)
            total_at_parent = parent_count if parent_count is not None else len(labels)
            return ["Leaf", label, count_label, total_at_parent]

        # base case 2: no attributes left to split
        if not available_attributes:
            majority_label = self._majority_vote(labels)
            count_majority = labels.count(majority_label)
            total_at_parent = parent_count if parent_count is not None else len(labels)
            return ["Leaf", majority_label, count_majority, total_at_parent]

        # recursive case: select best attribute and split
        best_att = self._select_attribute(instances, available_attributes)
        tree = ["Attribute", best_att]

        partitions = self._partition_instances(instances, best_att)
        # sort attribute values to make the tree deterministic
        values = sorted(partitions.keys())

        # attributes available to children (cannot reuse best_att)
        remaining_attributes = [a for a in available_attributes if a != best_att]

        # number of instances at this node becomes parent_count for children
        child_parent_count = len(instances)

        for val in values:
            subset = partitions[val]
            if not subset:
                # no instances at this branch -> majority vote at this node
                majority_label = self._majority_vote(labels)
                count_majority = labels.count(majority_label)
                subtree = ["Leaf", majority_label, count_majority, child_parent_count]
            else:
                subtree = self._tdidt(subset, remaining_attributes,
                                      parent_count=child_parent_count)

            tree.append(["Value", val, subtree])

        return tree
        
    def _collect_leaf_labels(self, subtree):
        """Collects all class labels under a subtree."""
        if subtree[0] == "Leaf":
            return [subtree[1]]
    
        labels = []
        for child in subtree[2:]:
            labels.extend(self._collect_leaf_labels(child[2]))
        return labels

    def _predict_instance(self, instance, subtree):
        """Traverses the decision tree for a single instance.

        Args:
            instance(list of obj): A single instance (feature values only,
                no class label).
            subtree(list): The nested list representation of the current
                subtree being traversed.

        Returns:
            obj: The predicted class label for the given instance.

        Notes:
            This helper is used by predict() to classify each instance in
            the test set.
        """
        node_type = subtree[0]
        if node_type == "Leaf":
            return subtree[1]
    
        # Attribute node
        att_index = subtree[1]    # integer index
        att_val = instance[att_index]
    
        # Look for matching child
        for value_branch in subtree[2:]:
            if value_branch[1] == att_val:
                return self._predict_instance(instance, value_branch[2])
    
        # Fallback: unseen attribute value -> majority vote among subtree leaves
        child_labels = []
        for value_branch in subtree[2:]:
            child_labels.extend(self._collect_leaf_labels(value_branch[2]))
    
        return self._majority_vote(child_labels)

    def _print_rules_recursive(self, subtree, conditions, attribute_names, class_name):
        """Performs a depth-first traversal to print decision rules.

        Args:
            subtree(list): The nested list representation of the current
                subtree in the decision tree.
            conditions(list of (int, obj)): A list of (attribute_index, value)
                pairs representing the path (conditions) from the root to
                the current node.
            attribute_names(list of str or None): A list of attribute names to
                use in the printed rules. If None, default names "att0",
                "att1", ... are used.
            class_name(str): The name of the target class to use in the
                printed rules.

        Returns:
            None

        Notes:
            This helper is used by print_decision_rules() to print rules in
            the format:
                IF att# == val AND ... THEN class_name = label
        """
        node_type = subtree[0]
    
        # Leaf node -> print rule
        if node_type == "Leaf":
            label = subtree[1]
            count = subtree[2]
            total = subtree[3]
    
            rule = "IF " + " AND ".join(conditions) + f" THEN {class_name} = {label}"
            rule += f" [Count: {count}/{total}]"
            print(rule)
            return
    
        # Attribute node
        att_index = subtree[1]   # integer attribute index
        att_name = attribute_names[att_index] if attribute_names else f"att{att_index}"
    
        # Explore branches
        for i in range(2, len(subtree)):
            value_branch = subtree[i]  # ["Value", val, child_subtree]
            branch_val = value_branch[1]
            new_condition = f"{att_name} == {branch_val}"
    
            self._print_rules_recursive(
                value_branch[2],
                conditions + [new_condition],
                attribute_names,
                class_name
            )

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        import subprocess
        # Build the DOT file
        with open(dot_fname, "w") as f:
            f.write("digraph DecisionTree {\n")
            f.write("    node [shape=box, style=filled, color=lightgray];\n")
    
            node_id = 0
    
            # Recursive helper
            def recurse(subtree, parent_id=None, label=None):
                nonlocal node_id
                this_id = node_id
                node_id += 1
    
                # Leaf
                if subtree[0] == "Leaf":
                    f.write(f'    node{this_id} [label="Leaf: {subtree[1]}"];\n')
                    if parent_id is not None:
                        f.write(f'    node{parent_id} -> node{this_id} [label="{label}"];\n')
                    return
    
                # Decision node
                attr_index = subtree[1]
                attr_name = attribute_names[attr_index] if attribute_names else f"att{attr_index}"
    
                f.write(f'    node{this_id} [label="{attr_name}"];\n')
                if parent_id is not None:
                    f.write(f'    node{parent_id} -> node{this_id} [label="{label}"];\n')
    
                # Each branch has form ["Value", value, child_subtree]
                for branch in subtree[2:]:
                    assert branch[0] == "Value"
                    value = branch[1]
                    child = branch[2]
                    recurse(child, this_id, value)
    
            recurse(self.tree)
            f.write("}\n")
    
        # Convert DOT → PDF
        try:
            subprocess.run(["dot", "-Tpdf", dot_fname, "-o", pdf_fname], check=True)
        except Exception as e:
            print("Graphviz error:", e)