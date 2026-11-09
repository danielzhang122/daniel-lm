# daniellm

i went down an ai rabbit hole and now i'm a little scared they're gonna take over the world. so i'm building one from scratch (somewhat). hopefully it'll be my in with the machines if things get weird.

in case the world actually ends, i'd like a quick little everything guide on hand, and preferably one that doesn't ask me to renew a subscription. i know i could just host a pre-trained llm locally, but wheres the fun in that? instead, i made a lobotomized model that kinda just spits out nonsense. so much for survival of the fittest.

**the goal:** an offline companion that can hold simple conversations and help find information in a saved reference library. It's trained on my own hardware and runs locally, with no subscription or internet needed. the tokenizer isn't my code: it's huggingface's `tokenizers` library, trained on my wikipedia data to learn its vocabulary. huggingface's `datasets` downloads the data, and pytorch's autograd does the backprop. (to be changed, see below)

i intend to eventually rewrite the backprop myself, or at least try to. i was told it would be a good exercise. to help teach me, i had claude generate me [this study guide](docs/llm-study-guide.md) in order to get me up to speed on how language models work at a lower level. essentially i'm using an llm to learn about llms so i can make an llm.

**the model:** a byte-level BPE tokenizer (huggingface's `tokenizers` library, with a vocabulary trained on this data) and a GPT-style decoder trained from random weights (no pretrained weights). About 34M parameters: 8 layers, width 512, 8 attention heads, 512-token context, 16,384-token vocabulary, tied input/output embeddings. Built for an RTX 4070 Ti (12 GB).

i also generated a really quick front-end to help me visualize training in real time. im finally putting my $3K gaming pc to use and doing something other than playing roblox.

tokenizers, transformers, neural networks, yada yada yada

## status

wikipedia pretraining is done, and right now it's just a dumb little ai: it repeats itself and makes things up, as you'd expect from 34m parameters. dialogue training, the offline library and the portable export are not finished.

## install (windows, python 3.13, nvidia gpu)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## quick start

```powershell
python prepare.py --articles 10000         # download and tokenize a small slice of Wikipedia
python train.py --steps 100                # short training run (output will be gibberish)
python generate.py --prompt 'The Solar System'
python dashboard.py                        # then open http://127.0.0.1:8765
```

`python` here means `.\.venv\Scripts\python.exe`.

for the full pipeline and to see what every file does, see [docs/usage.md](docs/usage.md).
