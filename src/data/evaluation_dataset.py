import torch
import random

import numpy as np

from torch.utils.data import Dataset

from src.data.mapping import map_id_to_vector
from src.data.preprocessing import pad_or_truncate


class EvaluationDataset(Dataset):
    def __init__(
        self,
        samples,
        num_negatives: int,
        max_sequence_length: int,
        padding_id: int = 0,
        mapping: dict = None,
        vector_size: int = 384,
    ):
        self.samples = samples
        self.max_sequence_length = max_sequence_length
        self.padding_id = padding_id
        if mapping is None:
            raise ValueError("mapping must contain the pre-tokenized article mapping")
        self.mapping = mapping
        self.num_negatives = num_negatives
        self.mapping_keys = self._mapping_keys()
        self.evaluation_samples = self._prepare_evaluation_samples()
        self.vector_size = vector_size
        
    def _mapping_keys(self):
        return list(self.mapping.keys())
        
    def _prepare_evaluation_samples(self):
        evaluation_samples = []
        for sample in self.samples:
            history = sample["history"]
            target = sample["target"]

            user_items = set(history)
            user_items.add(target)
            
            history = pad_or_truncate(
                sequence=history,
                max_sequence_length=self.max_sequence_length,
                padding_id=self.padding_id,
            )

            # Generate negative samples
            
            negative_samples = []
            while len(negative_samples) < self.num_negatives:
                negative_item = random.choice(self.mapping_keys)
                if negative_item not in user_items:
                    negative_samples.append(negative_item)

            evaluation_samples.append({
                # News IDs are strings (for example, "N1234"). Keep them as
                # IDs until map_id_to_vector converts them to numeric vectors.
                "history": history,
                "target": target,
                "negatives": negative_samples,
            })

        return evaluation_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        evaluation_sample = self.evaluation_samples[index]

        history = evaluation_sample["history"]
        candidate_ids = [evaluation_sample["target"]]
        candidate_ids.extend(evaluation_sample["negatives"])

        input_vectors = map_id_to_vector(
            sequence=history,
            mapping=self.mapping,
            vector_size=self.vector_size,
        )
        candidate_vectors = map_id_to_vector(
            sequence=candidate_ids,
            mapping=self.mapping,
            vector_size=self.vector_size,
        )

        return {
            "input_vectors": torch.from_numpy(
                np.asarray(input_vectors, dtype=np.float32)
            ),
            "candidate_vectors": torch.from_numpy(
                np.asarray(candidate_vectors, dtype=np.float32)
            ),
        }
