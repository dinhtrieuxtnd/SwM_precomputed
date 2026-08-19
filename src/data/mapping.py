from pyexpat import model

import pandas as pd

def collect_news_ids(samples: list[dict]) -> set[str]:
    news_ids: set[str] = set()

    for sample in samples:
        news_ids.update(sample["history"])
        news_ids.add(sample["target"])

    return news_ids


# Tạo mapping từ news_id sang text_input
# bằng cách kết hợp các cột được chỉ định
def build_mapping(
        df: pd.DataFrame,
        column_names: list
    ) -> dict:
    text_inputs = df[column_names].apply(
        lambda x: " - ".join(x.dropna().astype(str)),
        axis=1)
    mapping = dict(zip(df["news_id"], text_inputs))
    return mapping


def build_combined_mapping(
        train_news_df: pd.DataFrame,
        dev_news_df: pd.DataFrame,
        column_names: list
    ) -> dict:
    train_mapping = build_mapping(train_news_df, column_names)
    dev_mapping = build_mapping(dev_news_df, column_names)
    combined_mapping = {**train_mapping, **dev_mapping}
    return combined_mapping

def map_id_to_vector(
    sequence: list,
    mapping: dict,
    vector_size: int,
) -> list:
    padding_vector = [0] * vector_size
    return [ 
        mapping.get(news_id, padding_vector.copy())
        for news_id in sequence ]

def build_news_vector_mapping(
    mapping: dict,
    model,
) -> dict:
    keys = list(mapping)
    embeddings = model.encode(
        [mapping[key] for key in keys],
        batch_size=128
    )
    news_vector_mapping = dict(zip(keys, embeddings))
    return news_vector_mapping