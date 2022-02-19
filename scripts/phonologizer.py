from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens import Doc, Token
from spacy.training import Example, validate_get_examples
from spacy.util import registry, check_lexeme_norms
from spacy.vocab import Vocab
from thinc.api import Model, SequenceCategoricalCrossentropy
from thinc.types import Floats2d, Ints1d

# register custom phonemes property
Token.set_extension("phon", default=None)


def phon_score(examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
    return Scorer.score_token_attr(
        examples,
        attr="phon",
        getter=lambda t, attr: t._.get(attr),
        missing_values=set("_"),
        **kwargs,
    )


@registry.scorers("phon_scorer.v1")
def make_phon_scorer():
    return phon_score


class Phonologizer(TrainablePipe):
    """Pipeline component for grapheme-to-phoneme conversion."""

    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "phon",
        *,
        scorer=phon_score,
    ) -> None:
        """Initialize a grapheme-to-phoneme converter."""
        self.vocab = vocab
        self.model = model
        self.name = name
        cfg: Dict[str, Any] = {"labels": []}
        self.cfg = dict(sorted(cfg.items()))
        self.scorer = scorer

    @property
    def labels(self) -> Tuple[str, ...]:
        """Return the labels currently added to the pipe."""
        return tuple(self.cfg["labels"])

    def add_label(self, label: str) -> int:
        """Add a label to the pipe. Return 0 if label already exists, else 1."""
        if not isinstance(label, str):
            raise ValueError("Phonologizer labels must be strings")
        if label in self.labels:
            return 0
        self.cfg["labels"].append(label)
        self.vocab.strings.add(label)
        return 1

    def predict(self, docs: List[Doc]) -> List[Ints1d]:
        """Predict annotations for a batch of Docs, without modifying them."""
        # Handle cases where there are no tokens in any docs.
        if not any(len(doc) for doc in docs):
            n_labels = len(self.labels)
            guesses = [self.model.ops.alloc1i(n_labels) for _ in docs]
            return guesses

        # Get the scores and pick the highest-scoring guess for each token
        scores = self.model.predict(docs)
        assert len(scores) == len(docs), (len(scores), len(docs))
        guesses = [score.argmax(axis=1) for score in scores]
        assert len(guesses) == len(docs)
        return guesses

    def set_annotations(self, docs: Iterable[Doc], tag_ids: List[Ints1d]) -> None:
        """Annotate a batch of Docs, using pre-computed IDs."""
        labels = self.labels
        for doc, doc_tag_ids in zip(docs, tag_ids):
            for token, tag_id in zip(list(doc), doc_tag_ids):
                token._.phon = labels[tag_id]
        return docs

    def get_loss(
        self,
        examples: Iterable[Example],
        scores: List[Floats2d],
    ) -> Tuple[float, List[Floats2d]]:
        """Compute the loss and gradient for a batch of examples and scores."""
        # Create loss function
        loss_func = SequenceCategoricalCrossentropy(
            names=list(self.labels),
            normalize=False,
        )

        # Compute loss & gradient
        truths = self._examples_to_truth(examples)
        gradient, loss = loss_func(scores, truths)  # type: ignore
        if self.model.ops.xp.isnan(loss):
            raise ValueError(f"{self.name } loss is NaN")
        return loss, gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
    ):
        """Initialize the pipe for training using a set of examples."""
        validate_get_examples(get_examples, "Phonologizer.initialize")
        check_lexeme_norms(self.vocab, "phonologizer")

        # Read all unique tags from the examples and add them
        tags = set()
        for example in get_examples():
            for token in example.reference:
                if token._.phon:
                    tags.add(token._.phon)
        for tag in sorted(tags):
            self.add_label(tag)

        # Use the first 10 examples to sample Docs and labels
        doc_sample = []
        label_sample = []
        n_labels = len(self.labels)
        for example in islice(get_examples(), 10):
            doc_sample.append(example.reference)
            labeled = self._examples_to_truth([example])
            if labeled:
                label_sample += labeled
            else:
                label_sample.append(
                    self.model.ops.alloc2f(len(example.reference), n_labels)
                )

        # Initialize the model
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(
        self,
        examples: Iterable[Example],
    ) -> Optional[List[Floats2d]]:
        """Get the gold-standard labels for a batch of examples."""
        # Handle cases where there are no annotations in any examples
        tag_count = 0
        for example in examples:
            tag_count += len(
                list(filter(None, [token._.phon for token in example.reference]))
            )
        if tag_count == 0:
            return None

        # Get all the true labels
        truths = []
        for example in examples:
            gold_tags = self._get_aligned_phon(example)

            # Make a one-hot array for correct tag for each token
            gold_array = [
                [1.0 if tag == gold_tag else 0.0 for tag in self.labels]
                for gold_tag in gold_tags
            ]
            truths.append(self.model.ops.asarray2f(gold_array))  # type: ignore

        return truths

    def _get_aligned_phon(self, example: Example) -> List[Optional[str]]:
        """Get the aligned phonology data for a training Example."""
        align = example.alignment.x2y
        gold_ids = self.model.ops.asarray2f(
            [
                self.vocab.strings[tok._.phon] if tok._.phon else 0  # type: ignore
                for tok in example.reference
            ],
            dtype="uint64",
        )
        output = [None] * len(example.predicted)

        for token in example.predicted:
            if not token.is_alpha:
                output[token.i] = None
            else:
                id = gold_ids[align[token.i].dataXd].ravel()  # type: ignore
                if len(id) == 0 or id[0] == 0:
                    output[token.i] = None
                else:
                    output[token.i] = id[0]

        return [self.vocab.strings[id] if id else id for id in output]  # type: ignore


@Language.factory(
    "phonologizer",
    assigns=["token._.phon"],
    default_config={"scorer": {"@scorers": "phon_scorer.v1"}},
    default_score_weights={"phon_acc": 1.0},
)
def make_phonologizer(
    nlp: Language,
    model: Model[List[Doc], List[Floats2d]],
    name: str,
    scorer: Optional[Callable],
) -> Phonologizer:
    """Construct a Phonologizer component."""
    return Phonologizer(nlp.vocab, model, name=name, scorer=scorer)
