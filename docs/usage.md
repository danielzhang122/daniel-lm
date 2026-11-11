# daniellm usage

Full reference for running daniellm. The [README](../README.md) has the short version.

## Status

| Stage | State |
|---|---|
| Wikipedia pretraining, 34M model | Done (full-data stage, continued from the cleaned 100k-article experiment) |
| Wikipedia pretraining, 98M model | Running (`runs/big`) |
| FineWeb-Edu data (3B tokens) | Prepared at `E:\wiki-llm\data\fineweb-edu`, for a stage after `runs/big` |
| Wikipedia articles the 2023 snapshot lacks | Converted from the November 2026 dump: 963k articles, 0.8B tokens, at `E:\wiki-llm\data\wiki-missing` |
| FineWeb-Edu + those articles, combined | `E:\wiki-llm\data\fineweb-wiki` (3.76B train tokens), for the stage after `runs/big` |
| Dialogue training | Pipeline tested; lookup-style data (`dialogue-rag`) prepared; run it after pretraining finishes |
| Dashboard (charts, try the model, model explorer) | Working |
| Offline library search | Full index at `E:\wiki-llm\library.sqlite` (the 2023 snapshot plus the articles it lacked); `data/library.sqlite` is the 10,000-article starter |
| Chat that looks things up | Library passages go into the prompt ("Look it up in the library" on the Chat page); needs a model trained on `dialogue-rag` |
| Portable export for the Dell | Exported and tested locally; Linux install on the Dell not yet tested |

Expect limited ability from models this small: repetition, weak reasoning and made-up facts. Wikipedia pretraining alone doesn't teach conversation or factual accuracy. Treat it as a helper for reading references, not an authority.

## Files

| File | What it does |
|---|---|
| `model.py` | The GPT model |
| `prepare.py`, `cleaning.py` | Download and clean Wikipedia, train the tokenizer, write `train.bin` / `val.bin` |
| `prepare_fineweb.py` | Download FineWeb-Edu and pack it with an existing tokenizer, for a later stage |
| `prepare_wikidump.py` | Convert an official Wikipedia XML dump, e.g. to add articles the snapshot lacks |
| `combine_datasets.py` | Join prepared datasets that share a tokenizer into one |
| `val_loss.py` | Loss of a checkpoint on a dataset's validation split, on the same text every time |
| `eval_rag.py` | How well a checkpoint answers from passages (held-out SQuAD questions) |
| `train.py`, `training_utils.py` | Pretraining loop (resume, LR schedule, block sampler) |
| `train_full.py` | Shortcut for the full-Wikipedia training command |
| `generate.py` | Text generation from a checkpoint |
| `dialogue.py`, `prepare_dialogue.py`, `train_dialogue.py` | Conversation formatting and the dialogue training stage |
| `dashboard.py`, `dashboard/`, `inspect_model.py` | Local web UI |
| `library.py` | Offline Wikipedia search index |
| `evaluate.py`, `sample_store.py` | Save writing samples from checkpoints for comparison |
| `export_model.py`, `portable_server.py`, `verify_bundle.py` | Package the model to run offline on another machine |
| `bench_training.py` | Training speed benchmark |
| `tests/` | Unit tests |
| [llm-study-guide.md](llm-study-guide.md) | Beginner guide to how all of this works |

## Setup (PowerShell, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

`requirements-lock.txt` has the exact versions that were tested. Install it with `--extra-index-url https://download.pytorch.org/whl/cu128`.

The commands below use `python` to mean `.\.venv\Scripts\python.exe`.

## Pretraining

**Quick pilot** (checks the pipeline; the output will be mostly gibberish):
```powershell
python prepare.py --articles 10000
python train.py --steps 100
python generate.py --prompt 'The Solar System'
```

**Cleaned 100k-article experiment:**
```powershell
python prepare.py --articles 100000 --tokenizer-articles 20000 --out data/clean-100k
python train.py --data data/clean-100k --out runs/clean-100k --steps 5000 --eval-every 100
python generate.py --checkpoint runs/clean-100k/latest.pt --tokenizer data/clean-100k/tokenizer.json --prompt 'The Solar System' --tokens 150
```

**Full Wikipedia** (data at `E:\wiki-llm\data\full`, 4.5B tokens; one full pass is 550,550 steps, about 15 hours at the earlier benchmark):
```powershell
python prepare.py --articles 0 --tokenizer data/clean-100k/tokenizer.json --out E:\wiki-llm\data\full
python train_full.py                # start or resume
python train_full.py --steps 10000  # stop earlier to review progress
```
`train_full.py` loads the weights from `runs/clean-100k/latest.pt` and saves to `runs/full` every 500 steps. Ctrl+C stops it, and running the same command resumes from the last save.

**Bigger model** (`runs/big`, 98M parameters, about 60k tokens/s, one Wikipedia pass in about 21 hours):
```powershell
python train.py --data E:/wiki-llm/data/full --out runs/big --steps 275000 --schedule-steps 275000 --warmup 2000 --lr 0.0003 --min-lr 0.00003 --eval-every 1000 --batch 16 --accum 1 --context 1024 --width 768 --layers 12 --heads 12 --optimizer-backend fused --compile
```
To resume, run the same command with `--resume` added. The exact command is saved in `runs/big/command.txt`.

**FineWeb-Edu** (educational web text, for a later stage). It must reuse the tokenizer of the model you'll continue:
```powershell
$env:HF_HOME = 'E:\wiki-llm\hf-cache'   # keep the download cache off the nearly full C: drive
python prepare_fineweb.py --tokenizer E:/wiki-llm/data/full/tokenizer.json --out E:/wiki-llm/data/fineweb-edu --tokens 3000000000
```

**Articles the 2023 snapshot lacks** (for example "Paris"), from the official dump (about 25 GB from dumps.wikimedia.org; converting takes about 1.5 hours on 6 CPU cores):
```powershell
python prepare_wikidump.py --dump E:\wiki-llm\dump\enwiki-20261107-pages-articles-multistream.xml.bz2 --tokenizer E:/wiki-llm/data/full/tokenizer.json --skip-ids E:\wiki-llm\data\full\sources.jsonl --out E:/wiki-llm/data/wiki-missing
```
`--skip-ids` leaves out every article already in that dataset. Templates that carry article content (measurements such as `{{convert|330|m}}`, dates, foreign-language words) are kept as text ("330 m"). Infoboxes, references and tables are dropped.

**Combine datasets** (they must share a tokenizer; the trainer shuffles blocks, so the order doesn't matter):
```powershell
python combine_datasets.py E:/wiki-llm/data/fineweb-edu E:/wiki-llm/data/wiki-missing --out E:/wiki-llm/data/fineweb-wiki
```

**Compare checkpoints on the same text** (e.g. did a later stage make the model forget Wikipedia?):
```powershell
python val_loss.py --checkpoint runs/big/latest.pt --data E:/wiki-llm/data/full --blocks 64
```
It reads 64 evenly spaced 1,024-token blocks from `val.bin`, the same blocks every time, so two checkpoints are compared on exactly the same text. Lower is better. It runs on the CPU unless you add `--device cuda`.

**How training options work:**
- `--steps` is the total target, counting steps already done. It isn't "this many more".
- `--resume` continues the same run with its optimizer state and data position.
- `--init-from` copies only the weights into a new run, with a fresh optimizer and schedule. It needs the same tokenizer as the source run.
- Model size is set with `--width`, `--layers`, `--heads` and `--context` (defaults 512, 8, 8 and 512). With `--resume` or `--init-from` you can leave them out: they're read from the checkpoint, and a value that doesn't match is an error.
- `--compile` speeds up the model code with `torch.compile`. It needs `pip install "triton-windows<3.8"` on Windows. On this GPU it made no measurable difference, because the GPU was already 100% busy.
- Keep `--schedule-steps` the same whenever you resume.
- You can change `--batch` and `--accum` on resume as long as `batch × accum` stays the same.
- `prepare.py` refuses to write into a folder that isn't empty.
- Only load checkpoints you made yourself.

## Dashboard

```powershell
python dashboard.py
```
Then open **http://127.0.0.1:8765**. Its pages:
- **Training:** live loss charts and ready-to-copy commands.
- **Try the model:** text completion on the CPU from the latest checkpoint.
- **Model:** an architecture explorer.
- **Samples:** saved writing samples.
- **Library:** offline article search.
- **Chat:** only works with checkpoints from the dialogue stage.

The dashboard only reads logs and checkpoints. It never starts or changes training.

## Writing samples

```powershell
python evaluate.py --run full                              # once
python evaluate.py --run full --watch --interval 900       # every 15 min
```
This generates text from fixed prompts and saves it to `data/samples.sqlite`, so you can compare checkpoints on the Samples page. It's for judging by eye, not a benchmark.

## Offline library

```powershell
python library.py --articles 100000                                  # grow the index (resumable)
python library.py --db E:\wiki-llm\library.sqlite --articles 0       # full index
python library.py --data E:/wiki-llm/data/wiki-missing --db E:\wiki-llm\library.sqlite --articles 0   # add the dump articles
$env:DANIELLM_LIBRARY = 'E:\wiki-llm\library.sqlite'; python dashboard.py
```
It rebuilds the article text from the prepared token files, so no new download is needed. One library can hold articles from several datasets. Search results are passages from the saved articles; nothing is generated.

**Chat with lookup:** with "Look it up in the library" ticked, the dashboard searches the library for your question. From each top article it picks the paragraph sharing the most words with the question. The best article also contributes its opening paragraph, where Wikipedia states the key facts. Up to three passages go in front of the question. The reply lists the articles used. Only a model trained on passage + question examples (`dialogue-rag`, below) knows to answer from them.

## Dialogue stage (after pretraining)

```powershell
python prepare_dialogue.py --examples 5000
python train_dialogue.py --init-from runs/full/latest.pt --steps 1000
python train_dialogue.py --resume --steps 1000
```
**With lookup examples** (SQuAD 2.0, CC BY-SA 4.0: a Wikipedia paragraph, a question, and the answer marked in the paragraph. About a third of the questions can't be answered from their paragraph, which teaches the model to say "the passages don't say"):
```powershell
$env:HF_HOME = 'E:\wiki-llm\hf-cache'
python prepare_dialogue.py --tokenizer E:/wiki-llm/data/full/tokenizer.json --out E:/wiki-llm/data/dialogue-rag --examples 10000 --rag-examples 30000 --context 1024
```
With SQuAD's own third of unanswerable questions, the 98M model learned to refuse too often (it said "the passages don't say" for 55% of questions it could answer). `--rag-unanswerable 0.2` caps their share in the training split. The validation split keeps SQuAD's own mix, so scores stay comparable:
```powershell
python prepare_dialogue.py --tokenizer E:/wiki-llm/data/full/tokenizer.json --out E:/wiki-llm/data/dialogue-rag-v2 --examples 10000 --rag-examples 60000 --rag-val 1000 --rag-unanswerable 0.2 --context 1024
```

**Measure lookup answers** (CPU by default; add `--device cuda` only when nothing is training):
```powershell
python eval_rag.py --checkpoint runs/big-chat/latest.pt --questions 300
python eval_rag.py --checkpoint runs/big/latest.pt --questions 0      # answer loss only, to pick a starting checkpoint
```
- **Answer loss:** the loss on the answer tokens only, over every held-out SQuAD example (about 4 minutes on the CPU). It works for any checkpoint; lower is better.
- **For dialogue-trained checkpoints** it also has the model reply to each question, always picking the most likely next token so results repeat, and reports:
  - **answer found:** answerable questions whose reply contains a correct answer, ignoring case, punctuation and "a/an/the".
  - **right passage:** answerable questions whose reply cites the passage number the answer is in.
  - **says not found:** unanswerable questions whose reply says the passages don't say.
  - **false not found:** answerable questions where it wrongly says that.

Don't point it at the `latest.pt` of a run that's still training: copy the file right after a save and use the copy.

The general conversations come from [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), a synthetic dataset under the Apache-2.0 license. Only the assistant's replies count toward the loss. Running this alongside pretraining on the GPU may run out of memory. After training, open `/chat` in the dashboard.

## Export for offline use

```powershell
python export_model.py --checkpoint runs/full/latest.pt --tokenizer data/clean-100k/tokenizer.json --library data/library.sqlite --out exports/new-snapshot
python exports/new-snapshot/verify_bundle.py
```
Use a new `--out` folder each time. The bundle contains the weights, tokenizer, code, library and checksums. It doesn't include Python or the Linux packages: build those on the target machine (the export's own README has the steps).

## Data

- Source: the November 1, 2023 English Wikipedia snapshot ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)).
- Cleaning cuts each article at sections like References, See also and External links.
- About 1% of articles are held out for validation, chosen by a hash of the article ID.
- Keep `sources.jsonl` and `metadata.json` with any data built from it, and check the attribution/share-alike terms before sharing.
- Keep `.venv`, `data`, `.cache` and `runs` out of OneDrive.
