from pathlib import Path
from typing import Optional

import typer
import wandb
from spacy import util
from spacy.cli import _util
from spacy.training.initialize import init_nlp
from spacy.training.loop import train
from thinc.api import Config


def main(default_config: Path, output_path: Path, code: Optional[Path] = None):
    loaded_local_config = util.load_config(default_config)

    # Allow importing from an external script, to load the Phonologizer
    _util.import_code(code)
    
    with wandb.init() as run:
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        nlp = init_nlp(merged_config)
        output_path.mkdir(parents=True, exist_ok=True)
        train(nlp, output_path, use_gpu=True)


if __name__ == "__main__":
    typer.run(main)
