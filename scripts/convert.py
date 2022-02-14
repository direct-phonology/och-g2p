#!/usr/bin/env python3

import random
from pathlib import Path

import spacy
import typer
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def convert(in_dir: Path, out_dir: Path, tokens_per_doc: int, test_size: float) -> None:
    """Convert g2p training data into spaCy's format for training."""

    # load the spaCy model and create a container for generated docs
    nlp = spacy.blank("och")
    docs = []

    # set the custom extension for phonological features
    spacy.tokens.Token.set_extension("phon", default=None)

    # iterate over the input directory
    for file in tqdm(in_dir.glob("*.txt")):
        annotations = [line.split("\t") for line in file.read_text().splitlines()]

        # create a new doc every tokens_per_doc characters
        while len(annotations) >= tokens_per_doc:
            tokens = [annotations.pop() for _ in range(0, tokens_per_doc)]

            # ensure we only get docs of size tokens_per_doc
            if len(tokens) < tokens_per_doc:
                continue

            # create a new doc
            doc = spacy.tokens.Doc(
                nlp.vocab,
                words=[token[0] for token in tokens],
                spaces=[False] * tokens_per_doc,
            )

            # set the phonological features on each token
            for t in doc:
                if tokens[t.i][1] != "_":
                    t._.phon = tokens[t.i][1]

            # store it
            docs.append(doc)
            annotations = annotations[:-tokens_per_doc]

    # split the docs into train/dev sets
    random.shuffle(docs)
    train_docs, rest = train_test_split(docs, test_size=test_size)
    dev_docs, test_docs = train_test_split(rest, test_size=test_size)

    # save the output in spacy's format
    train_db = spacy.tokens.DocBin(store_user_data=True)
    for doc in train_docs:
        train_db.add(doc)
    train_db.to_disk(out_dir / "train.spacy")

    dev_db = spacy.tokens.DocBin(store_user_data=True)
    for doc in dev_docs:
        dev_db.add(doc)
    dev_db.to_disk(out_dir / "dev.spacy")

    test_db = spacy.tokens.DocBin(store_user_data=True)
    for doc in test_docs:
        test_db.add(doc)
    test_db.to_disk(out_dir / "test.spacy")


if __name__ == "__main__":
    typer.run(convert)

__doc__ = convert.__doc__  # type: ignore
