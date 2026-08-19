from pathlib import Path

import pandas as pd


BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def read_behaviors(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep="\t",
        names=BEHAVIOR_COLUMNS,
    )


def read_news(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep="\t",
        names=NEWS_COLUMNS,
    )


def get_clicked_news(impressions: str) -> list[str]:
    clicked_news = []
    
    for impression in impressions.split():
        news_id, label = impression.rsplit("-", 1)

        if label == "1":
            clicked_news.append(news_id)

    return clicked_news