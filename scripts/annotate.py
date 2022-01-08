#!/usr/bin/env python3

import re
from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin


def annotate(in_dir: Path, out_dir: Path) -> None:
    """Generate and save g2p training data for spaCy."""

    nlp = spacy.blank("och")

    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        for start, label in annotations:
            doc[start : start + 1]._.label = label
        db.add(doc)

    db.to_disk(out_dir / "train.spacy")


if __name__ == "__main__":
    typer.run(annotate)

__doc__ = annotate.__doc__  # type: ignore
