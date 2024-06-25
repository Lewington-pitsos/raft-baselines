from collections import Counter
import random
from typing import Mapping, Optional

import datasets

from raft_baselines.classifiers.classifier import Classifier

class PluralityClassifier(Classifier):
    def __init__(self, training_data: datasets.Dataset, **kwargs) -> None:
        super().__init__(training_data)
        self.most_frequent_class = self._find_most_frequent_class(training_data)

    def _find_most_frequent_class(self, training_data: datasets.Dataset) -> str:
        class_counts = {c: 0.0 for c in self.classes}
        for example in training_data:
            class_counts[self.classes[example['Label'] - 1]] += 1
        
        most_frequent_class = max(class_counts, key=class_counts.get)
        print('most frequent class', most_frequent_class)


        return most_frequent_class

    def classify(
        self,
        target: Mapping[str, str],
        random_seed: Optional[int] = None,
        should_print_prompt: bool = False,
    ) -> Mapping[str, float]:
        result = {c: 0.0 for c in self.classes}
        result[self.most_frequent_class] = 1.0

        return result