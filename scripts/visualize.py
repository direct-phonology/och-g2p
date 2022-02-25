from typing import Optional
from pathlib import Path

import spacy
import streamlit as st
import pandas as pd
import typer

from spacy_och.examples import sentences


def main(model: Path, code: Optional[Path] = None) -> None:
    st.title("Grapheme-to-phoneme testing")

    spacy.cli._util.import_code(code)

    nlp = spacy.load(model)
    docs = [nlp(sent) for sent in sentences]
    data = [[str(token._.phon) for token in doc] for doc in docs]
    df = pd.DataFrame(data)

    st.dataframe(df)


if __name__ == "__main__":
    typer.run(main)
