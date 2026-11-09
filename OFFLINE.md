# Offline tools and the next training stage

The running Wikipedia job stays separate from these tools. They do not change its weights or optimizer.

## What works now

- **Training / Try:** existing live metrics and CPU text completion.
- **Samples:** generated text is stored in `data/samples.sqlite`; filter by prompt to compare checkpoints. These are output snapshots, not recoverable model checkpoints. The latest 200 are shown; older records stay in the database.
- **Library:** search and read a starter index of 10,000 cleaned Wikipedia articles without internet. Source URLs are retained. Results are retrieved passages, not generated answers. The index is partial and not a complete offline Wikipedia mirror.
- **Chat:** conversation formatting, context trimming, CPU generation and UI are implemented. The UI accepts only dialogue-stage checkpoints. No useful dialogue-trained model is ready yet.
- **Portable snapshot:** `exports/daniellm-offline` contains an intermediate CPU inference model, tokenizer, code, local web server, reference database and checksums. It does not contain Python, Linux wheels, or a finished conversational model. See its README for installation and recovery steps.

## Save comparable writing samples

```powershell
.\.venv\Scripts\python.exe evaluate.py --run full
# In a separate terminal, evaluate new checkpoints at most once every 15 minutes:
.\.venv\Scripts\python.exe evaluate.py --run full --watch --interval 900
```

Uses three fixed prompts, seed 42, temperature 0.8 and a 60-token output limit. Each record reports the actual checkpoint step loaded; a checkpoint can change between prompts. Re-run after substantial training and compare topic relevance, sentence completion, repetition and factual errors. This is qualitative evaluation, not a validated knowledge benchmark. CPU inference still shares system RAM and I/O with training. Do not run multiple watchers.

Setup started one background watcher. Its logs are `runs/sample-watch.log` and `runs/sample-watch-errors.log`. It lasts until stopped or the PC restarts; it is not an installed startup service. To stop that watcher without stopping training, run this in PowerShell:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*wiki-llm*evaluate.py*--watch*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
```

## Grow the offline library

```powershell
# Continue the existing starter library to 100,000 articles:
.\.venv\Scripts\python.exe library.py --articles 100000
# Or build the full library on E: (additional disk and indexing time required):
.\.venv\Scripts\python.exe library.py --db E:\wiki-llm\library.sqlite --articles 0
# Point the dashboard at that database before starting it:
$env:DANIELLM_LIBRARY = 'E:\wiki-llm\library.sqlite'
.\.venv\Scripts\python.exe dashboard.py
```

The index reconstructs article text from existing packed tokens and source IDs; it needs no new download. It checks title alignment before attributing an article. Progress is committed every 250 articles, so rerunning the same command resumes after interruption. `--articles` is a total target, not an additional count. A new `--db` starts a new index. Keep the input metadata, tokenizer, binary tokens and `sources.jsonl` together while indexing. Normal search needs only the SQLite database. Keep original source/license information when redistributing text: https://huggingface.co/datasets/wikimedia/wikipedia . The original reference sections were removed during training preparation, so this is a reading/search aid rather than a faithful complete mirror.

## Dialogue training after Wikipedia pretraining

The prepared starter dataset has 4,910 training and 90 validation conversations from [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk). The provider labels it synthetic and Apache-2.0; its source card and exact revision are saved in `data/dialogue`. Inspect source/component terms before redistribution. This is a starter set, not a complete reviewed corpus for a reliable assistant.

We keep complete conversations that fit 513 tokens (512 next-token positions), mask user/system text out of the loss, and train on assistant answers plus the existing EOS marker. Prompt hashes separate train/validation and remove duplicate prompt sequences. This does not eliminate all semantic near-duplicates. Plain role labels reuse the existing tokenizer without changing its vocabulary.

```powershell
# Already prepared; use another --out if you want a larger set:
.\.venv\Scripts\python.exe prepare_dialogue.py --examples 5000
# AFTER the Wikipedia run finishes, start a separate dialogue model:
.\.venv\Scripts\python.exe train_dialogue.py --init-from runs/full/latest.pt --steps 1000
# Resume that dialogue stage with unchanged schedule settings:
.\.venv\Scripts\python.exe train_dialogue.py --resume --steps 1000
```

The dialogue trainer uses full precision and defaults to CUDA; it is intentionally separate from pretraining. It checks tokenizer/context compatibility, saves resumable checkpoints every 100 steps and on completion, and records metrics for the dashboard. `--steps` is a total stage target. The initial `--schedule-steps 1000` stays fixed on resume. Running another GPU trainer during pretraining can exhaust memory; the verification here used CPU and an isolated `data/dialogue-smoke` output for only three steps. Those smoke-test weights are not a useful chat model and are not exposed as a production run.

After a real dialogue run, open `/chat` and select `dialogue`. It uses a fixed 80-token reply budget, trims oldest complete turns, and keeps history only in the browser tab. It has no persistent personal memory. Opening a new tab or clearing the conversation resets it. Review output quality before assuming it can answer questions reliably.

## Export and verify a later model

```powershell
.\.venv\Scripts\python.exe export_model.py --checkpoint runs/full/latest.pt --tokenizer data/clean-100k/tokenizer.json --library data/library.sqlite --out exports/new-snapshot
.\.venv\Scripts\python.exe exports/new-snapshot/verify_bundle.py
```

Use a new output folder each time. The exporter removes optimizer/RNG training state and retains model architecture, tokenizer identity, phase and step. SQLite backup creates a consistent library copy. Copy the whole export folder to the Dell. Its web server currently offers text completion and source search; dialogue-trained bundles also support the shared `generate.py --messages-json` CLI format.

The export works offline once compatible dependencies are installed. Create dependency wheels on the target Linux architecture/Python version; the Windows virtual environment cannot be copied as a Linux installation. Actual Dell CPU speed, Linux setup and recovery from saved wheels remain untested. Back up the model, tokenizer, code, references, Python/OS installers and Linux wheels independently. Checksums detect corruption but do not authenticate an untrusted bundle.

## Remaining work

Finish pretraining, run and evaluate dialogue training, expand/review the reference corpus, test source-grounded answers and abstention, and test the Dell installation. A persistent desktop avatar with backed-up personal memories is a future idea, not implemented here. None of these tools proves the model is ready for high-stakes advice.

## Verification performed

Fourteen unit tests passed, including assistant-label masking, conversation trimming, library indexing/resume, sample persistence and corrupted-export detection. Real HTTP checks passed for the pages, search, article reading, sample saving and CPU generation. A pretraining checkpoint was correctly rejected by the chat endpoint. The dialogue trainer completed two CPU steps and resumed to step three in an isolated output folder. The portable web server generated text and searched its included library; exported generation also ran with Python socket connections blocked. Browser visual/interaction automation and execution on Linux hardware were unavailable, so those remain unverified.
