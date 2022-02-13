from itertools import islice
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.scorer import Scorer
from spacy.tokens import Doc, Token
from spacy.training import Example
from spacy.vocab import Vocab
from thinc.api import (
    Model,
    Optimizer,
    SequenceCategoricalCrossentropy,
    set_dropout_rate,
)
from thinc.types import Floats2d, Ints1d

# register custom phonemes property
Token.set_extension("phon", default=None)


class Phonologizer(TrainablePipe):
    """Pipeline component for grapheme-to-phoneme conversion."""

    def __init__(self, vocab: Vocab, model: Model, name: str = "phon") -> None:
        """Initialize a grapheme-to-phoneme converter."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self._labels: List[str] = []

    @property
    def labels(self) -> Tuple[str, ...]:
        """Returns the labels currently added to the pipe."""
        return tuple(self._labels)

    def add_label(self, label: str) -> int:
        """Add a label to the pipe. Return 0 if label already exists, else 1."""
        if not isinstance(label, str):
            raise ValueError("Phonologizer labels must be strings")
        if label in self._labels:
            return 0
        self._labels.append(label)
        self.vocab.strings.add(label)
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the model to a Doc, set annotations, and return it."""
        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> List[Ints1d]:
        """Predict annotations for a batch of Docs, without modifying them."""
        # Handle cases where there are no tokens in any docs.
        if not any(len(doc) for doc in docs):
            n_labels = len(self.labels)
            guesses: List[Ints1d] = [self.model.ops.alloc((0, n_labels)) for _ in docs]
            return guesses

        # Get the scores and pick the highest-scoring guess for each token
        scores = self.model.predict(docs)
        guesses = [score.argmax(axis=1) for score in scores]
        return guesses

    def set_annotations(self, docs: Iterable[Doc], tag_ids: List[Ints1d]) -> None:
        """Annotate a batch of Docs, using pre-computed scores."""
        for doc, doc_tag_ids in zip(docs, tag_ids):
            for token, tag_id in zip(list(doc), doc_tag_ids):
                token._.phon = self.vocab.strings[self._labels[tag_id]]

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> dict[str, float]:
        """Learn from training examples and update the model."""
        # Initialize loss tracking if not set up
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)

        # Handle cases where there are no tokens in any docs
        if not any(
            len(example.predicted) if example.predicted else 0 for example in examples
        ):
            return losses

        # Set dropout rate
        set_dropout_rate(self.model, drop)

        # Compute loss & gradient; do backprop
        scores, backprop = self.model.begin_update(
            [example.predicted for example in examples]
        )
        for score in scores:
            if self.model.ops.xp.isnan(score.sum()):
                raise ValueError("Score sum is NaN")
        loss, gradient = self.get_loss(examples, scores)
        backprop(gradient)
        if sgd:
            self.finish_update(sgd)

        # Add loss to tracked total
        losses[self.name] += loss
        return losses

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
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training using a set of examples."""
        # Add preset labels if requested
        if labels:
            for label in labels:
                self.add_label(label)

        # Otherwise read all unique tags from the examples and add them
        else:
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
        for example in islice(get_examples(), 10):
            doc_sample.append(example.reference)
            doc_truths = self._examples_to_truth([example])
            if doc_truths:
                label_sample.append(self.model.ops.asarray(doc_truths, dtype="float32"))

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
            gold_tags = [token._.phon for token in example.reference]

            # Make a one-hot array for correct tag for each token
            gold_array = [
                [1.0 if tag == gold_tag else 0.0 for tag in self.labels]
                for gold_tag in gold_tags
            ]
            truths.append(self.model.ops.asarray(gold_array, dtype="float32"))

        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, float]:
        """Score a batch of examples."""
        return Scorer.score_token_attr(
            examples,
            attr="phon",
            getter=lambda t, attr: t._.get(attr),
            **kwargs,
        )


@Language.factory(
    "phonologizer",
    assigns=["token._.phon"],
    default_score_weights={"phon_acc": 1.0},
)
def make_phonologizer(
    nlp: Language,
    model: Model[List[Doc], List[Floats2d]],
    name: str,
) -> Phonologizer:
    """Construct a Phonologizer component."""
    return Phonologizer(nlp.vocab, model, name=name)
