<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Grapheme-to-Phoneme Conversion for Old Chinese

This project applies [Baxter and Sagart's (2014)](https://ocbaxtersagart.lsait.lsa.umich.edu/) reconstruction of Old Chinese phonology to the pronounciations given in the [_Jingdian Shiwen_](https://en.wikipedia.org/wiki/Jingdian_Shiwen) in order to build a model for converting Old Chinese graphemes-in-context into phonemes.

By linking the Middle Chinese _fanqie_ pronounciations to Old Chinese equivalents using Baxter and Sagart's system, we automatically annotate training data for the model, which disambiguates polyphonic characters based on context.

Chinese text used is from the [Kanseki Repository](https://www.kanripo.org/text/KR1g0003/) and is licensed CC-BY-SA 4.0.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `install` | Install dependencies |
| `parse` | Reformat data used for annotation |
| `annotate` | Generate training data |
| `convert` | Convert training data into spaCy's format |
| `train` | Train a grapheme-to-phoneme conversion model |
| `evaluate` | Test the model's performance |
| `package` | Package the model so it can be installed |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `install` &rarr; `parse` &rarr; `annotate` &rarr; `train` &rarr; `evaluate` &rarr; `package` |
| `tune` | `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/jdsw-org` | Git | Text of the _Jingdian Shiwen_ by Lu Deming, from Kanripo |
| `assets/kr-norm.tsv` | URL | Normalization table for special entities in Kanripo |
| `assets/kr-unicode.tsv` | URL | Unicode conversion table for special entities in Kanripo |
| `assets/sbgy.xml` | URL | XML version of the Song Ben Guang Yun rhyme dictionary |
| `assets/baxter-sagart-oc.xlsx` | URL | Baxter and Sagart's (2014) reconstruction of Old Chinese phonology |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->
