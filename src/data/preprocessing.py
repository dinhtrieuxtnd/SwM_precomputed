import random
import re

import pandas as pd

from src.data.parser import get_clicked_news

# Tạo các sample từ DataFrame behaviors
# Mỗi sample là một dict với các key: "history" và "target"
def build_samples_id(df: pd.DataFrame) -> list[dict]:
    samples = []

    for _, row in df.iterrows():
        if pd.isna(row["history"]):
            continue

        history = row["history"].split()
        clicked_news = get_clicked_news(row["impressions"])

        for target in clicked_news:
            samples.append({
                "history": history,
                "target": target,
            })

    return samples

def pad_or_truncate(
    sequence: list,
    max_sequence_length: int,
    padding_id: int = 0,
) -> list:
    if len(sequence) > max_sequence_length:
        sequence = sequence[-max_sequence_length:]

    padding_length = max_sequence_length - len(sequence)

    return [padding_id] * padding_length + sequence



# Xóa các ký tự đặc biệt và khoảng trắng thừa trong text
# Không loại bỏ dấu câu (. , !, ?, : ; ...) vì chúng có thể mang ý nghĩa trong ngữ cảnh
# dấu nháy, dấu gạch nối (', -) cũng được giữ lại vì chúng có thể xuất hiện trong các từ ghép hoặc tên riêng
# Ký hiện tiền tệ, phần trăm ($, %, ...) cũng được giữ lại vì chúng có thể xuất hiện trong các bài viết về kinh tế, tài chính
def clean_text(text: str) -> str:
    # Loại bỏ các ký tự không phải chữ cái, số, dấu câu, dấu nháy, dấu gạch nối, ký hiệu tiền tệ và phần trăm
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-/$%]", "", text)

    # Thay thế nhiều khoảng trắng liên tiếp bằng một khoảng trắng duy nhất
    text = re.sub(r"\s+", " ", text)

    # Loại bỏ khoảng trắng ở đầu và cuối chuỗi
    text = text.strip()

    return text

def clean_news_text(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = df[column].apply(clean_text)
    return df

def build_training_pair(
    history: list,
    target,
    max_sequence_length: int,
    padding_id: int = 0,
    
)-> tuple[list, list]:

    sequence = history + [target]
    
    # Input là tất cả các phần tử của sequence ngoại trừ phần tử cuối cùng, [A, B, C, D] -> [A, B, C]
    input_sequence = sequence[:-1]
    # Positive là tất cả các phần tử của sequence ngoại trừ phần tử đầu tiên. [A, B, C, D] -> [B, C, D]
    positive_sequence = sequence[1:]
    
    input_sequence = pad_or_truncate(
        sequence=input_sequence,
        max_sequence_length=max_sequence_length,
        padding_id=padding_id,
    )
    
    positive_sequence = pad_or_truncate(
        sequence=positive_sequence,
        max_sequence_length=max_sequence_length,
        padding_id=padding_id,
    )

    return input_sequence, positive_sequence    
        
def sample_negative_item(
    user_items: set,
    mapping_keys: list,
) -> int:
    while True:
        negative_item = random.choice(mapping_keys)
        if negative_item not in user_items:
            return negative_item
        
def build_negative_sequence(
    positive_sequence: list,
    user_items: set,
    mapping_keys: list,
    padding_id: int = 0,
) -> list:
    negative_sequence = []

    for positive_item in positive_sequence:
        if positive_item == padding_id:
            negative_sequence.append(padding_id)
            continue
        negative_item = sample_negative_item(   
            user_items=user_items,
            mapping_keys=mapping_keys,
        )
        negative_sequence.append(negative_item)

    return negative_sequence