"""Data functions"""
import json
import re

from datasets import Dataset


def clean(answer: str):
    return re.sub(r"<<.*?>>", "", answer).replace("####", "The answer is")


def read_jsonl(path: str):
    with open(path) as fh:
        load = [json.loads(line) for line in fh.readlines() if line]
    for d in load:
        d["answer"] = clean(d["answer"])
    return load


def load_data(train: str = "./grade-school-math/grade_school_math/data/train.jsonl",
              test: str = "grade-school-math/grade_school_math/data/test.jsonl"):
    train_data = Dataset.from_list(read_jsonl(train))
    test_data = Dataset.from_list(read_jsonl(test))
    return train_data, test_data
