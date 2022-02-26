from typing import Optional
from pathlib import Path

import spacy
import streamlit as st
import pandas as pd
import typer

from spacy_och.examples import sentences


def main(model: Path, code: Optional[Path] = None) -> None:
    spacy.cli._util.import_code(code) # type: ignore
    nlp = spacy.load(model)
    
    st.set_page_config(layout="wide")
    st.title("Grapheme-to-phoneme testing")

    st.header("Examples")
    docs = [nlp(sent) for sent in sentences]
    for doc in docs:
        data = (
            [token.text for token in doc],
            [token._.phon for token in doc],
        )
        st.table(pd.DataFrame(data, index=("char", "read")))

    st.header("Live test")
    st.text("TODO")


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        pass
