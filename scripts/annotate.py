#!/usr/bin/env python3

import re
from pathlib import Path
from collections import Counter

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

MULTI = re.compile(r"""
    ^
    $
""", re.VERBOSE | re.MULTILINE)
# fmt: on

# case where there's no annotation yet
EMPTY_ANNO = re.compile(r"^(.)\t_$", re.MULTILINE)

# types of annotations we care about
WHITELIST = (FANQIE, DIRECT)

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


def annotation_to_mc(annotation: re.Match, mc_table: pd.DataFrame) -> str:
    """Convert annotations into Middle Chinese readings."""

    STATS["total"] += 1

    # fanqie annotation: fetch initial and rime, combine, then choose the
    # reading for the annotated character that matches the combination
    fanqie = FANQIE.match(annotation.group(0))
    if fanqie:
        initial = mc_table[mc_table["zi"] == fanqie.group("initial")]
        rime = mc_table[mc_table["zi"] == fanqie.group("rime")]

        if initial.empty:
            # typer.echo(f"No MC reading for {fanqie.group('initial')} (initial)")
            STATS["missing_total"] += 1
            STATS["no_reading"][fanqie.group("initial")] += 1
            return f"{fanqie.group('char')}\t_"

        if rime.empty:
            # typer.echo(f"No MC reading for {fanqie.group('rime')} (rime)")
            STATS["missing_total"] += 1
            STATS["no_reading"][fanqie.group("rime")] += 1
            return f"{fanqie.group('char')}\t_"

        reading = "".join(
            (initial["MCInitial"].iloc[0], rime["MCfinal"].iloc[0])
        ).replace("-", "")

        STATS["annotated_total"] += 1

        return f"{fanqie.group('char')}\t{reading}"

    # direct "sounds like" annotation: fetch the reading directly, then choose
    # the reading for the annotated character that matches it
    direct = DIRECT.match(annotation.group(0))
    if direct:
        match = mc_table[mc_table["zi"] == direct.group("anno")]

        if match.empty:
            # typer.echo(f"No MC reading for {direct.group('anno')} (direct)")
            STATS["missing_total"] += 1
            STATS["no_reading"][direct.group("anno")] += 1
            return f"{direct.group('char')}\t_"

        if len(match) > 1:
            # typer.echo(f"Character {direct.group('anno')} is polyphonic")
            STATS["missing_total"] += 1
            STATS["polyphonic"][direct.group("anno")] += 1
            return f"{direct.group('char')}\t_"

        reading = match["MC"].iloc[0]

        STATS["annotated_total"] += 1

        return f"{direct.group('char')}\t{reading}"

    # empty annotations: see if the character is monophonic, and if so, just
    # use the reading for that character
    monophone = EMPTY_ANNO.match(annotation.group(0))
    if monophone:
        STATS["initially_empty"] += 1
        match = mc_table[mc_table["zi"] == monophone.group(1)]

        if match.empty:
            STATS["missing_total"] += 1
            STATS["no_reading"][monophone.group(1)] += 1
            return f"{monophone.group(1)}\t_"

        if len(match) > 1:
            STATS["missing_total"] += 1
            STATS["polyphonic"][monophone.group(1)] += 1
            return f"{monophone.group(1)}\t_"

        reading = match["MC"].iloc[0]

        STATS["annotated_total"] += 1

        return f"{monophone.group(1)}\t{reading}"

    # shouldn't get here
    raise UserWarning(f"Bad annotation format: {annotation.group(0)}")


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
    _annotation_to_mc = lambda char: annotation_to_mc(char, mc_table)

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

        # convert annotations into middle chinese readings
        text = ANNOTATION.sub(_annotation_to_mc, text)

        # add middle chinese readings for any remaining non-polyphones
        text = EMPTY_ANNO.sub(_annotation_to_mc, text)

        # save the text into the output folder
        output = mc_dir / f"{file.stem}.txt"
        output.open(mode="w").write(text)

    # process middle chinese readings into old chinese readings
    typer.echo("Converting to old chinese annotations...")
    for file in tqdm(sorted(list(mc_dir.glob("*.txt")))):

        # add middle chinese readings for any remaining non-polyphones
        text = EMPTY_ANNO.sub(_mc_to_oc, file.read_text())

        # save the text into the output folder
        output = oc_dir / f"{file.stem}.txt"
        output.open(mode="w").write(text)

    # print statistics
    typer.echo("\nStatistics:")
    typer.echo(f"  {STATS['total']} total characters")
    typer.echo(f"  {STATS['initially_empty']} characters initially empty")
    typer.echo(f"  {STATS['annotated_total']} annotated characters")
    typer.echo(f"  {STATS['missing_total']} unannotated characters")
    typer.echo(f"  {STATS['polyphonic'].total()} polyphonic characters")
    typer.echo(f"  Top 5:\t\t{STATS['polyphonic'].most_common(5)}")
    typer.echo(f"  {STATS['no_reading'].total()} characters with no reading")
    typer.echo(f"  Top 5:\t\t{STATS['no_reading'].most_common(5)}")

    # write out statistics
    polyphone_stats = Path("polyphones.txt")
    polyphone_stats.open(mode="w").write(
        "\n".join(
            [
                "\t".join([entry[0], str(entry[1])])
                for entry in STATS["polyphonic"].most_common(100)
            ]
        )
    )
    no_reading_stats = Path("no_reading.txt")
    no_reading_stats.open(mode="w").write(
        "\n".join(
            [
                "\t".join([entry[0], str(entry[1])])
                for entry in STATS["no_reading"].most_common(100)
            ]
        )
    )


if __name__ == "__main__":
    typer.run(annotate)

__doc__ = annotate.__doc__  # type: ignore
