# Caption Generator script

The `caption_generator.py` script is a utility for generating descriptive captions and speech quality evaluations using Google Gemini models. It supports both MOS (Mean Opinion Score) prediction for individual speech samples and A/B preference testing between paired samples.

## Prerequisites

Before running the script, you must have a valid Google Gemini API key.

### Setting the API Key

The script requires the `GEMINI_API_KEY` environment variable to be set. You can set this in your terminal session before running the script:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

To make this persistent, add the export command to your shell configuration file (e.g., `~/.zshrc` or `~/.bashrc`).

## Usage

The script is executed via the `uv run` command.

### Basic Command

Running with default settings (processes MOS dataset):

```bash
uv run src/asa/caption_generator.py
```

### Command-line Arguments

The script accepts the following arguments to specify input and output files:

| Argument | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `data/processed/mos_dataset.json` | Path to the input JSON file containing speech metadata. |
| `--output` | `-o` | `data/processed/mos_predictions.json` | Path where the results with generated captions will be saved. |
| `--help` | `-h` | N/A | Show the help message and exit. |

### Examples

**Process a custom MOS dataset:**

```bash
uv run src/asa/caption_generator.py --input data/my_dataset.json --output data/my_results.json
```

**Process an A/B test dataset:**

```bash
uv run src/asa/caption_generator.py --input data/processed/ab_dataset.json --output data/processed/ab_predictions.json
```

## Functionality

### MOS Prediction

When the input JSON contains individual utterances with `meta` fields (mos, noi, col, dis, loud), the script generates a descriptive evaluation of the speech quality.

**Input Structure Example:**
```json
{
  "utt_id": "sample1.wav",
  "audio_path": "path/to/audio.wav",
  "meta": {
    "mos": 3.5,
    "noi": 4.0,
    ...
  }
}
```

**Output Structure Example:**
```json
{
  "utt_id": "sample1.wav",
  "audios": ["path/to/audio.wav"],
  "response": "The speech has good quality with minor background noise...",
  "query": "Please describe and evaluate the synthetic speech<audio>.",
  "mos": 3.5,
  "noi": 4.0,
  ...
}
```

### A/B Testing

When the input JSON contains paired utterances (`meta_a` and `meta_b`), the script performs an A/B preference test to determine which sample is better based on the metadata.

**Input Structure Example:**
```json
{
  "pair_id": "pair1",
  "audio_a_path": "path/to/audio_a.wav",
  "audio_b_path": "path/to/audio_b.wav",
  "meta_a": { ... },
  "meta_b": { ... },
  "winner": "A"
}
```

**Output Structure Example:**
```json
{
  "pair_id": "pair1",
  "audios": ["path/to/audio_a.wav", "path/to/audio_b.wav"],
  "response": "Speech A is better due to lower distortion...",
  "query": "Please perform A/B preference test between<audio>and<audio>, including a tie.",
  "winner_predicted": "A",
  "A_mos": 3.5,
  "B_mos": 2.8,
  ...
}
```
