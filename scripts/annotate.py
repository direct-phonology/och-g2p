#!/usr/bin/env python3

import re
from pathlib import Path
from collections import Counter
from typing import cast

import pandas as pd
import typer
from tqdm import tqdm

# fmt: off
ANNOTATION = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    (?P<anno>[^_]+?)        # annotation
    $
""", re.VERBOSE | re.MULTILINE)

FANQIE = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    (?P<initial>[^_])       # initial/onset
    (?P<rime>[^_])          # rime/tone
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

MULTI_FANQIE = re.compile(r"""
    ^
    (?P<char>.)             # character being annotated
    \t
    (?P<initial>[^_])       # initial/onset
    (?P<rime>[^_])          # rime/tone
    反下同
    $
""", re.VERBOSE | re.MULTILINE)
# fmt: on

# case where there's no annotation yet
EMPTY_ANNO = re.compile(r"^(?P<char>.)\t_$", re.MULTILINE)

# types of annotations we care about
WHITELIST = (FANQIE, DIRECT, MULTI_FANQIE)

# track statistics for analysis
STATS = {
    "total": 0,
    "annotated_total": 0,
    "missing_total": 0,
    "initially_empty": 0,
    "no_reading": Counter(),
    "polyphonic": Counter(),
}


def filter_annotation(annotation: re.Match) -> str:
    """Filter out annotations that we can't convert into readings."""

    # if the annotation isn't in the whitelist, turn it into a _ (blank)
    if not any(annotype.match(annotation.group(0)) for annotype in WHITELIST):
        return f"{annotation.group('char')}\t_"

    return annotation.group(0)


def fanqie_to_mc(annotation: re.Match, mc_table: pd.DataFrame) -> str:
    """Convert a fanqie annotation into a Middle Chinese reading."""
    STATS["total"] += 1  # type: ignore

    # fanqie annotation: fetch initial and rime, combine, then choose the
    # reading for the annotated character that matches the combination
    initial = mc_table[mc_table["zi"] == annotation.group("initial")]
    rime = mc_table[mc_table["zi"] == annotation.group("rime")]

    if initial.empty:
        STATS["missing_total"] += 1  # type: ignore
        STATS["no_reading"][annotation.group("initial")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    if rime.empty:
        STATS["missing_total"] += 1  # type: ignore
        STATS["no_reading"][annotation.group("rime")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    reading = "".join((initial["MCInitial"].iloc[0], rime["MCfinal"].iloc[0])).replace(
        "-", ""
    )

    STATS["annotated_total"] += 1  # type: ignore

    return f"{annotation.group('char')}\t{reading}"


def direct_to_mc(annotation: re.Match, mc_table: pd.DataFrame) -> str:
    """Convert a 'sounds like' annotation into a Middle Chinese reading."""
    STATS["total"] += 1  # type: ignore

    # direct "sounds like" annotation: fetch the reading directly, then choose
    # the reading for the annotated character that matches it
    match = mc_table[mc_table["zi"] == annotation.group("anno")]

    if match.empty:
        STATS["missing_total"] += 1  # type: ignore
        STATS["no_reading"][annotation.group("anno")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    if len(match) > 1:
        STATS["missing_total"] += 1  # type: ignore
        STATS["polyphonic"][annotation.group("anno")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    reading = match["MC"].iloc[0]

    STATS["annotated_total"] += 1  # type: ignore

    return f"{annotation.group('char')}\t{reading}"


def missing_to_mc(annotation: re.Match, mc_table: pd.DataFrame) -> str:
    """Add Middle Chinese readings for un-annotated characters."""
    STATS["total"] += 1  # type: ignore
    STATS["initially_empty"] += 1  # type: ignore

    # empty annotations: see if the character is monophonic, and if so, just
    # use the reading for that character
    match = mc_table[mc_table["zi"] == annotation.group("char")]

    if match.empty:
        STATS["missing_total"] += 1  # type: ignore
        STATS["no_reading"][annotation.group("char")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    if len(match) > 1:
        STATS["missing_total"] += 1  # type: ignore
        STATS["polyphonic"][annotation.group("char")] += 1  # type: ignore
        return f"{annotation.group('char')}\t_"

    reading = match["MC"].iloc[0]

    STATS["annotated_total"] += 1  # type: ignore

    return f"{annotation.group('char')}\t{reading}"


def multi_fanqie_to_mc(txt: str, mc_table: pd.DataFrame) -> str:
    """Convert all multi-fanqie annotations in a text to readings."""

    #     find the position of the first non-blank annotation at position `j > i` for character `X`
    #     find all blank annotations for character `X` at positions `i < k < j`
    #     change all blank annotations to the annotation `A`
    #     change the initial "all below" annotation to annotation `A`

    # while there are "all below" annotations:
    for annotation in MULTI_FANQIE.finditer(txt):

        # find the next such annotation in the file
        # annotation = MULTI_FANQIE.search(txt)  # type: ignore
        char = annotation.group('char') # type: ignore
        initial = mc_table[mc_table["zi"] == annotation.group("initial")]  # type: ignore
        rime = mc_table[mc_table["zi"] == annotation.group("rime")]  # type: ignore

        if initial.empty:
            STATS["missing_total"] += 1  # type: ignore
            STATS["no_reading"][annotation.group("initial")] += 1  # type: ignore
            continue

        if rime.empty:
            STATS["missing_total"] += 1  # type: ignore
            STATS["no_reading"][annotation.group("rime")] += 1  # type: ignore
            continue

        reading = "".join((initial["MCInitial"].iloc[0], rime["MCfinal"].iloc[0])).replace(
            "-", ""
        )

        # find the position of the next non-blank annotation for this character
        next_anno = re.compile(f"^{char}\t[^_]+?$").search(txt, pos=annotation.end())   # type: ignore
        stop_pos = next_anno.start() if next_anno else len(txt)

        # find all blank annotations in between and add the reading to them
        blank_annos = re.compile(f"^{char}\t_$") \
                        .findall(txt, pos=annotation.start(), endpos=stop_pos) # type: ignore
        # for anno in blank_annos:
        #     txt[anno.start:anno.end] = f"{char}\t{reading}"
        STATS["total"] += len(blank_annos) + 1 
        STATS["annotated_total"] += len(blank_annos) + 1  # type: ignore

        # change the initial "all below" annotation to the reading
        txt = txt[:annotation.start()] + f"{char}\t{reading}" + txt[annotation.end():]  # type: ignore

    return txt


def mc_to_oc(char: re.Match, oc_table: pd.DataFrame) -> str:
    """Convert a Middle Chinese reading into an Old Chinese reading."""
    return char.group(0)


def annotate(
    in_dir: Path, mc_dir: Path, oc_dir: Path, mc_table_path: Path, oc_table_path: Path
) -> None:
    """Convert the Jingdian Shiwen annotations into Old Chinese readings."""

    # read baxter's song ben guang yun middle chinese table
    mc_table = pd.read_excel(
        mc_table_path, usecols=["zi", "MC", "MCInitial", "MCfinal"]
    ).drop_duplicates()
    _fanqie_to_mc = lambda char: fanqie_to_mc(char, mc_table)
    _direct_to_mc = lambda char: direct_to_mc(char, mc_table)
    _missing_to_mc = lambda char: missing_to_mc(char, mc_table)
    _multi_fanqie_to_mc = lambda txt: multi_fanqie_to_mc(txt, mc_table)

    # read baxter & sagart's old chinese table
    oc_table = pd.read_excel(
        oc_table_path, usecols=["zi", "MC", "OC"]
    ).drop_duplicates()
    _mc_to_oc = lambda char: mc_to_oc(char, oc_table)

    # clean out destination directories
    for loc in [mc_dir, oc_dir]:
        loc.mkdir(exist_ok=True)
        for file in loc.glob("*.txt"):
            file.unlink()

    # add middle chinese readings for as many characters as possible
    typer.echo("Processing middle chinese annotations...")
    for file in tqdm(sorted(list(in_dir.glob("*.txt")))):

        # read the file and strip annotations we can't convert
        text = ANNOTATION.sub(filter_annotation, file.read_text())

        # convert "below all the same" multi-annotations
        # text = _multi_fanqie_to_mc(text)

        # convert fanqie annotations
        text = FANQIE.sub(_fanqie_to_mc, text)

        # convert direct "sounds like" annotations
        text = DIRECT.sub(_direct_to_mc, text)

        # add middle chinese readings for any remaining non-polyphones
        text = EMPTY_ANNO.sub(_missing_to_mc, text)

        # save the text into the output folder
        output = mc_dir / f"{file.stem}.txt"
        output.open(mode="w").write(text)

    # print statistics
    typer.echo("\nStatistics:")
    typer.echo(f"  {STATS['total']} total characters")
    typer.echo(f"  {STATS['initially_empty']} characters initially empty")
    typer.echo(f"  {STATS['annotated_total']} annotated characters")
    typer.echo(f"  {STATS['missing_total']} unannotated characters")
    typer.echo(f"  {STATS['polyphonic'].total()} polyphonic characters")  # type: ignore
    typer.echo(f"  Top 5:\t\t{STATS['polyphonic'].most_common(5)}")  # type: ignore
    typer.echo(f"  {STATS['no_reading'].total()} characters with no reading")  # type: ignore
    typer.echo(f"  Top 5:\t\t{STATS['no_reading'].most_common(5)}")  # type: ignore

    # write out statistics
    polyphone_stats = Path("polyphones.txt")
    polyphone_stats.open(mode="w").write(
        "\n".join(
            [
                "\t".join([entry[0], str(entry[1])])
                for entry in STATS["polyphonic"].most_common(100)  # type: ignore
            ]
        )
    )
    no_reading_stats = Path("no_reading.txt")
    no_reading_stats.open(mode="w").write(
        "\n".join(
            [
                "\t".join([entry[0], str(entry[1])])
                for entry in STATS["no_reading"].most_common(100)  # type: ignore
            ]
        )
    )


if __name__ == "__main__":
    typer.run(annotate)

__doc__ = annotate.__doc__  # type: ignore
