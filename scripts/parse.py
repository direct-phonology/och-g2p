#!/usr/bin/env python3

import re
from pathlib import Path
from typing import Callable

import pandas as pd
import typer


# texts we don't care about using: paratextual material, notes, etc.
BLACKLIST = [
    "KR1g0003_000",  # introduction
    "KR1g0003_001",  # 序録
    "KR1g0003_031",  # notes
    "KR1g0003_032",  # notes
    "KR1g0003_033",  # notes
]

# artifacts in kanseki repository text
MODE_HEADER = re.compile(r"# -\*- mode: .+; -\*-\n")
META_HEADER = re.compile(r"#\+\w+: .+\n")
LINE_BREAK = re.compile(r"<pb:(?:.+)>¶\n")

# fmt: off
KanripoEntity = re.compile(r"""
    (
    &(?P<id>KR\d+);     # entity with a code/id, e.g. &KR0001;
    |
    (?P<combo>\[.+?\])  # entity defined as a combo, e.g. [阿-可+利]
    )
""", re.VERBOSE)
# fmt: on

# fmt: off
TextWithAnno = re.compile(r"""
    (?P<text>.+?)   # text preceding an annotation
    \(              # opening parenthesis
    (?P<anno>.+?)   # annotation text
    \)              # closing parenthesis
""", re.VERBOSE)
# fmt: on


def split_text_anno(match: re.Match) -> str:
    """Split text preceding an annotation up by character."""
    return "".join(
        [
            "\t_\n".join(list(match.group("text"))),
            "\t",
            match.group("anno"),
            "\n",
        ]
    )


def krp_entity_unicode(table: pd.DataFrame, match: re.Match) -> str:
    """Private use unicode representation for a Kanripo entity."""

    # entity will be either an id or a combo, we don't care which
    entity = match.group("id") or match.group("combo")

    # fetch from the table; warn if not found
    char = table.loc[table["form"] == entity]
    if char.empty:
        raise UserWarning(f"Kanripo entity not found: {entity}")

    return char["unicode"].values[0]


def clean_text(text: str, to_unicode: Callable[[re.Match], str]) -> str:
    """Clean an org-mode text and convert entities into unicode."""

    # replace kanripo entities with unicode placeholders
    text = KanripoEntity.sub(to_unicode, text)

    # strip headers, page breaks, newlines, etc.
    text = MODE_HEADER.sub("", text)
    text = META_HEADER.sub("", text)
    text = LINE_BREAK.sub("\n", text)
    text = text.replace("¶", "")
    text = "".join(text.strip().splitlines())

    # some annotations were split across lines; we need to recombine them
    text = text.replace(")(", "")

    # remove all remaining whitespace
    text = "".join(text.split())

    return text


def split_text(text: str) -> str:
    """Reformat a text into a CoNLL-U-like format."""
    return TextWithAnno.sub(split_text_anno, text)


def parse(src_dir: Path, txt_dir: Path, unicode_table: Path) -> None:
    """Transform the Jingdian Shiwen text into raw text and annotations."""

    # read unicode conversion table
    unicode_table = pd.read_csv(
        "assets/kr-unicode.tsv",
        sep="\t",
        names=["form", "unicode"],
    )
    to_unicode = lambda entity: krp_entity_unicode(unicode_table, entity)

    # clean out destination directory
    txt_dir.mkdir(exist_ok=True)
    for file in txt_dir.glob("*.txt"):
        file.unlink()

    # process source text
    for file in sorted(list(src_dir.glob("*.txt"))):

        # ignore blacklisted material
        if any([file.stem in name for name in BLACKLIST]):
            continue

        # read the file and clean it
        text = clean_text(file.read_text(), to_unicode)

        # reformat conll-u style
        text = split_text(text)

        # save the text into the raw text folder
        output = txt_dir / f"{file.stem}.txt"
        output.open(mode="w").write(text)


if __name__ == "__main__":
    typer.run(parse)

__doc__ = parse.__doc__  # type: ignore
