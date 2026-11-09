# daniellm

i went down an AI rabbit hole and now i'm a little scared they're gonna take over the world. so i'm building this guy from scratch. hopefully my in with the machines if things get weird.

also, if the world ends, i'd like a little post-apocalyptic everything guide on hand. preferably one that doesn't ask me to renew my subscription while i'm living in a shed.

**The actual goal:** a somewhat reliable offline companion that can hold simple conversations and help me find and understand information in a saved reference library. Trained on my own hardware, hosted locally, with no API subscription or internet needed for everyday use. That's the goal, not what the current model can do yet. It will still need a working computer and power.

Own byte-level BPE tokenizer and randomly initialized GPT-style decoder. No pretrained model weights. Defaults: about 34M parameters, 8 layers, width 512, 8 attention heads, 512-token context, tied input/output embeddings, and a 16,384-token vocabulary. Intended for an RTX 4070 Ti 12 GB.

## Roadmap to conversation

1. **Wikipedia pretraining ? running.** The full-data stage continues the cleaned experiment's weights. Track loss and read saved samples as it learns.
2. **Dialogue stage ? pipeline tested, useful model pending.** A short-conversation starter set is prepared. Assistant-only training, checkpoint resume, shared role formatting and context trimming passed CPU smoke tests. Run the substantive stage after pretraining.
3. **Interface ? implemented.** Training metrics, text completion, saved sample comparisons, and offline library search work locally. The chat page activates only for dialogue-stage checkpoints; no useful conversational checkpoint has been trained yet.
4. **Offline references ? partial starter index ready.** Search 10,000 reconstructed cleaned articles and read their source text. The resumable indexer can process the rest. Generated, source-grounded answers and reliability evaluation remain future work.
5. **Portable inference ? exported and locally tested.** An intermediate weights-only snapshot includes source code, tokenizer, reference library and checksums. Linux dependency packaging and testing on the actual Dell still need doing.

See [OFFLINE.md](OFFLINE.md) for commands, verification results, limitations, and the next-stage workflow.

Success means basic back-and-forth exchanges, relevant replies to simple prompts, correct turn boundaries, and some use of recent context. Evaluate those behaviors with a fixed set of prompts alongside held-out dialogue loss. There is no single loss threshold that proves conversational ability.

For the reliability goal, also test factual questions against the saved sources, verify that cited passages support the answers, and check whether the assistant admits when information is missing. Record errors and limitations instead of treating fluent output as evidence of accuracy. It should help consult references, not serve as the sole authority for high-stakes decisions. Offline source search is implemented; generated source-grounded answers and systematic reliability evaluations are not yet implemented.

Expect limited capability from 34M parameters: repetition, weak reasoning, and invented facts may remain. The current 512-token context must accommodate the prompt, recent history, and response; long conversations will need trimming. Wikipedia pretraining alone does not teach reliable conversation or factual accuracy. A larger model or longer context is a separate experiment, not part of the current training run.

## Local training dashboard

Open a second PowerShell terminal:

```powershell
cd C:\path\to\daniel-lm
.\.venv\Scripts\python.exe dashboard.py
```

Visit **http://127.0.0.1:8765**. daniellm refreshes your actual logs every three seconds, charts training/validation loss, explains the next steps, and builds copyable commands for the selected experiment. Training commands run only when you paste them into a terminal. The **Try the model** page generates real text directly in your browser using the latest saved checkpoint. No extra packages or cloud hosting are required. Last log time indicates freshness, not proof that a process is running. Keep the dashboard terminal open; Ctrl+C stops it.

Open **Model** in the dashboard for an interactive graph of sampled real feed-forward weights, plus the selected checkpoint's embeddings, transformer blocks, dimensions, and parameter counts. Enter text to inspect its real tokenizer output and click a token to explore the causal attention mask. The diagram shows the network structure and allowed visibility; it does not show actual attention weights or live activations. Inspection loads a trusted local checkpoint on CPU and does not train or modify it.

### Try the current model while training

Visit **http://127.0.0.1:8765/play** or click **Try the model**. Choose a run, write the beginning of a paragraph, and click Generate. Results show the actual saved checkpoint step, token count, and generation time. Run names and prompts are sent only to the local server.

Generation uses CPU with two PyTorch threads, leaving GPU memory for training. It still shares CPU, RAM, and disk resources. One request runs at a time, with a 200-token UI limit and a 120-second timeout. The model is loaded fresh for each request so newly saved checkpoints become available without restarting the dashboard. A fixed sampling seed makes same-setting comparisons easier. This is independent text completion, not conversation memory; repetition and invented facts are expected at this stage.

If the dashboard was already running when this feature was installed, stop **only its terminal** with Ctrl+C and rerun `.\.venv\Scripts\python.exe dashboard.py`. Keep the training terminal running.

The cleaned dataset scanned 100,000 articles and retained 86,968 after filtering, with 53,728,494 training tokens and 546,776 validation tokens. It has its own tokenizer and model; the pilot is preserved.

The cleaned run completed 1,000 steps, including a successful resume from step 100. Validation loss went from 7.3703 to 5.6000; throughput was approximately 85,831 tokens/second. Continue with the `--resume --steps 5000` command below or copy the command from the dashboard. These losses are not directly comparable to the pilot's different tokenizer/data. Dashboard HTTP endpoints and backend tests passed; browser visual verification was unavailable.

## Verified first run (November 8, 2026)

- CUDA PyTorch 2.11.0+cu128 on the RTX 4070 Ti; BF16 enabled.
- Tokenizer trained from scratch; 10,000 articles prepared into 5,708,154 training tokens and 51,999 validation tokens (111 held-out articles).
- 33,854,464 model parameters; 100 training steps completed, then successfully resumed to step 101.
- Validation loss: 7.1417 at step 50, 6.7859 at step 100, 6.7788 at step 101.
- Initial 100-step run averaged about 84,421 tokens/second with 0.90 GiB peak PyTorch tensor allocation (not total device memory).
- Causal attention and simple learning tests passed; dependency check passed; generation from the saved checkpoint passed. Output remains mostly incoherent, as expected after only 827,392 sampled training tokens.
- Installed versions are recorded in `requirements-lock.txt`. For reproduction, install that file with `--extra-index-url https://download.pytorch.org/whl/cu128`.

The pilot data and checkpoint already exist. Use the continuation command below rather than rerunning preparation into `data/pilot`.

## Setup (PowerShell)

Run commands from this directory. The environment uses Python 3.13.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest test_model
```

## First experiment

```powershell
.\.venv\Scripts\python.exe prepare.py --articles 10000
.\.venv\Scripts\python.exe train.py --steps 100
.\.venv\Scripts\python.exe generate.py --prompt 'The Solar System'
```

100 steps is a pipeline and speed check, not a trained language model. Early output will mostly be gibberish. Training on encyclopedia prose learns text completion; it does not automatically teach chat or reliable question answering.

The source is the **November 1, 2023 English Wikipedia snapshot**, not live Wikipedia. Streaming avoids downloading the entire raw corpus upfront. Preparation stores packed uint16 tokens (2 bytes per token), article attribution URLs, and the exact dataset revision. The tokenizer learns only from training articles. A deterministic article-ID hash reserves roughly 1% of articles for validation before tokenization. Related/duplicate articles can still cross splits; validation is a learning signal, not a rigorous knowledge benchmark.

## Continue and scale

### Full Wikipedia: continue the existing model

The prepared dataset at `E:\wiki-llm\data\full` contains 4,510,104,716 training tokens. A full pass takes 550,550 optimizer steps at the default batch/accumulation/context settings. The `full` stage transfers weights from the cleaned experiment's step-5,000 checkpoint, checks tokenizer identity and architecture, and starts a new optimizer, sampler and schedule at step zero. Stage steps do not include earlier training. The source checkpoint is preserved.

```powershell
# Start or resume the full pass (roughly 15 hours at the earlier benchmark):
.\.venv\Scripts\python.exe train_full.py
# Or stop at a smaller total stage target to review progress first:
.\.venv\Scripts\python.exe train_full.py --steps 10000
```

The script stores checkpoints in this project's `runs/full` so the local dashboard discovers the run automatically; the dataset stays on E:. Select `full` in the dashboard. Checkpoints save every 500 steps, and at the requested final step. Leave the training terminal open and prevent the PC from sleeping. Ctrl+C stops training; rerun the same command to resume the last saved checkpoint. Work since that checkpoint is repeated. Keep the schedule horizon at 550,550 even for shorter stages.

The equivalent continuation command is:

```powershell
.\.venv\Scripts\python.exe train.py --data E:\wiki-llm\data\full --out runs/full --resume --steps 550550 --schedule-steps 550550 --lr 0.0001 --min-lr 0.00001 --eval-every 500
```

For a **new output directory**, replace `--resume` with `--init-from runs/clean-100k/latest.pt`. Transfer loads weights only; ordinary resume restores the optimizer and data position. They cannot be combined. Token counts and tokenizer hashes are checked before training. One pass covers the shuffled training blocks (the final partial block wraps to the beginning); validation articles are never trained on. This is a coverage target, not a guarantee of model quality.

The cleaned experiment scans 100,000 articles and drops articles with fewer than 200 characters after cleanup. It trains a new tokenizer on up to 20,000 retained training articles, so it starts a **fresh model** in a separate folder. Your original pilot checkpoint stays in `runs/pilot`.

```powershell
# Prepare once (skip if data/clean-100k already exists):
.\.venv\Scripts\python.exe prepare.py --articles 100000 --tokenizer-articles 20000 --out data/clean-100k
# First bounded run:
.\.venv\Scripts\python.exe train.py --data data/clean-100k --out runs/clean-100k --steps 1000 --eval-every 100
# Continue the same model, keeping the schedule horizon fixed:
.\.venv\Scripts\python.exe train.py --data data/clean-100k --out runs/clean-100k --resume --steps 5000 --eval-every 100
.\.venv\Scripts\python.exe generate.py --checkpoint runs/clean-100k/latest.pt --tokenizer data/clean-100k/tokenizer.json --prompt 'The Solar System' --tokens 150
# Later, prepare all English articles using the same vocabulary:
.\.venv\Scripts\python.exe prepare.py --articles 0 --tokenizer data/clean-100k/tokenizer.json --out data/full
.\.venv\Scripts\python.exe train.py --data data/full --out runs/full --steps 10000
```

`--steps` is the total target, including resumed steps. `--resume` requires the same data metadata and model configuration; new checkpoints also require matching batch, accumulation and schedule settings. Only load your own trusted checkpoint files. Checkpoints include optimizer and random-generator states and replace `latest.pt` atomically. Interruptions lose steps since the latest checkpoint (every 50 steps by default, 100 in these commands).

Full-data commands above start a new model. The pilot is deliberately disposable. For a better vocabulary, train a fresh tokenizer on a larger sample before starting a serious run. The data preparer refuses nonempty output folders; interrupted preparation must use a fresh output directory.

New runs use shuffled blocks without replacement: every block is visited once per epoch, with the final partial block wrapping to the start. Epoch order is deterministic, and resuming reconstructs the position from the completed steps. Limited-step runs may stop before a full epoch. Old pilot checkpoints retain their original random-window sampling and constant learning rate when resumed.

New runs warm up for 200 steps to 0.0003, then follow cosine decay to 0.00003 at step 10,000. `--schedule-steps` is independent of the stopping point `--steps`: use the same horizon on resume. Beyond that horizon the rate stays at its minimum. Stop and review validation and generation before repeatedly extending training.

Cleanup stops at an exact standalone References, Bibliography, External links, See also, or similar heading, removes explicit category lines, and collapses excess blank lines. This is a heuristic: useful later sections can be lost, and unlabeled category text may remain. It is applied identically to tokenizer training, model training, and validation; source URLs and cleaning settings are retained. Loss values from different tokenizers/datasets should not be directly compared.

Use the run's `metrics.jsonl` to estimate time: `target tokens / measured tokens per second`. The default optimizer step processes 8,192 tokens. The logged average includes validation/checkpoint overhead after the first evaluation. Reduce `--batch` to 1 before starting a fresh run if VRAM is tight.

C: had about 39 GiB free at setup; E: had about 690 GiB. Before full preparation, choose an appropriate local data location with `--out`, and point training at it with `--data`. Avoid syncing `.venv`, `data`, `.cache`, and `runs` through OneDrive: `.gitignore` only affects Git. Network/model downloads are cached under this project's `.cache`.

Dataset source and licensing information: https://huggingface.co/datasets/wikimedia/wikipedia . Keep `sources.jsonl` and the dataset metadata with derived datasets; consult the source's attribution/share-alike terms before redistribution.

References: https://huggingface.co/docs/datasets/stream , https://huggingface.co/docs/tokenizers/quicktour , https://pytorch.org/get-started/locally/ .
