#!/usr/bin/env python3

import re
from pathlib import Path

import pandas as pd
import typer

# fmt: off
ANNOTATION = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    (?P<anno>[^_]+?)       # annotation
    $
""", re.VERBOSE | re.MULTILINE)

FANQIE = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    (?P<A>[^_])             # fanqie A
    (?P<B>[^_])             # fanqie B
    反
    $
""", re.VERBOSE | re.MULTILINE)

DIRECT = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    音
    (?P<anno>[^_])          # sounds "the same as" this character
    $
""", re.VERBOSE | re.MULTILINE)
# fmt: on


# types of annotations we care about
WHITELIST = (FANQIE, DIRECT)


def filter_annotation(annotation: re.Match) -> str:
    """Filter out annotations that we can't convert into readings."""

    # if the annotation isn't in the whitelist, turn it into a _ (blank)
    if not any(annotype.match(annotation.group(0)) for annotype in WHITELIST):
        return f"{annotation.group('char')}\t_"

    return annotation.group(0)


def annotate(in_dir: Path, out_dir: Path, bs_table: Path) -> None:
    """Convert the Jingdian Shiwen annotations into Old Chinese readings."""

    # read baxter-sagart reconstruction table
    bs = pd.read_excel(bs_table)

    # clean out destination directory
    out_dir.mkdir(exist_ok=True)
    for file in out_dir.glob("*.txt"):
        file.unlink()

    # process conll-style annotations
    for file in sorted(list(in_dir.glob("*.txt"))):

        # read the file and strip annotations we can't convert
        text = ANNOTATION.sub(filter_annotation, file.read_text())

        # convert annotations into middle chinese readings

        # add middle chinese readings from baxter/schuessler for non-polyphones

        # convert all readings from middle chinese into old chinese

        # save the text into the output folder
        output = out_dir / f"{file.stem}.txt"
        output.open(mode="w").write(text)


if __name__ == "__main__":
    typer.run(annotate)

__doc__ = annotate.__doc__  # type: ignore
