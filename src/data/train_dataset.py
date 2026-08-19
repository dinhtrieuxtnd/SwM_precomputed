import numpy as np
import torch

from torch.utils.data import Dataset
from src.data.mapping import map_id_to_vector
from src.data.preprocessing import (
    build_training_pair,
    build_negative_sequence,
)

class TrainDataset(Dataset):

    def __init__(
        self,
        samples,
        max_sequence_length: int,
        padding_id: int = 0,
        mapping: dict = None,
        vector_size: int = 384,
    ):
        self.samples = samples
        self.max_sequence_length = max_sequence_length
        self.padding_id = padding_id
        self.vector_size = vector_size
        if mapping is None:
            raise ValueError("mapping must contain the pre-tokenized article mapping")
        self.mapping = mapping
        self.mapping_keys = list(self.mapping.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        history = sample["history"]
        target = sample["target"]

        input_ids, positive_ids = (
            build_training_pair(
                history=history,
                target=target,
                max_sequence_length=self.max_sequence_length,
                padding_id=self.padding_id,
            )
        )

        user_items = set(history)
        user_items.add(target)

        
        negative_ids = build_negative_sequence(
            positive_sequence=positive_ids,
            user_items=user_items,
            mapping_keys=self.mapping_keys,
            padding_id=self.padding_id,
        )
        
        input_vectors = map_id_to_vector(
            sequence=input_ids,
            mapping=self.mapping,
            vector_size=self.vector_size,
        )
        positive_vectors = map_id_to_vector(
            sequence=positive_ids,
            mapping=self.mapping,
            vector_size=self.vector_size,
        )
        negative_vectors = map_id_to_vector(
            sequence=negative_ids,
            mapping=self.mapping,
            vector_size=self.vector_size,
        )

        return {
            "input_vectors": torch.from_numpy(
                np.asarray(input_vectors, dtype=np.float32)
            ),
            "positive_vectors": torch.from_numpy(
                np.asarray(positive_vectors, dtype=np.float32)
            ),
            "negative_vectors": torch.from_numpy(
                np.asarray(negative_vectors, dtype=np.float32)
            ),
        }
