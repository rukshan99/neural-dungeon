"""The books of the Library of Echoes.

A small, fixed corpus about the dungeon itself: its floors, bosses, loot and
rules. Every trial on Floor 9 retrieves from these passages, so they are
deliberately short (two to four sentences) and deliberately imperfect:

* a few passages are NEAR-DUPLICATES of others (the same fact, reworded),
* one is a WHITESPACE-ONLY duplicate (for the deduplication room),
* a few are DECOYS that share vocabulary with a topic but mean something else
  (the refectory's "chunks", the Librarian's "index" finger, a late-fee "rate").

``GOLDEN_QA`` holds twelve questions whose answers appear *verbatim* in exactly
one passage. ``OFF_TOPIC_QUESTIONS`` are questions the library cannot answer;
a grounded reader must refuse them.

Nothing here is a stub. Import it from rooms, solutions and trials alike::

    from floors.floor_09_library_of_echoes.assets.corpus import PASSAGES, GOLDEN_QA
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QA:
    question: str
    answer: str  # appears verbatim in exactly one passage
    passage_id: str


def _p(pid: str, title: str, text: str) -> Passage:
    return Passage(pid, title, text)


PASSAGES: tuple[Passage, ...] = (
    # ------------------------------------------------------------ the floors
    _p("p01", "The Threshold",
       "Floor 0 is the Threshold, where every spell is spoken in shapes. Its four rooms "
       "teach arrays, reshaping, broadcasting and vectorization. The floor needs only numpy."),
    _p("p02", "The Broadcasting Basilisk",
       "The Broadcasting Basilisk guards the exit of Floor 0. Its gaze petrifies numpy's "
       "broadcasting helpers, so travellers must apply the four broadcasting rules to plain "
       "shape tuples by hand. Its weakness is someone who knows the rules by heart."),
    _p("p03", "The Caverns of Descent",
       "Floor 1 is the Caverns of Descent. The gauge on every traveller's wrist reads loss, "
       "and every corridor slopes down. The rooms teach loss functions, gradients, gradient "
       "checking, gradient descent, momentum, Adam and learning-rate schedules."),
    _p("p04", "The Learning-Rate Lich",
       "The Learning-Rate Lich rules the narrowest valley of Floor 1. Step boldly and it hurls "
       "you up the far wall; step timidly and you grow old on its slope. Its weakness is "
       "a learning rate that changes over time."),
    _p("p05", "The Chain of Whispers",
       "Floor 2 is the Chain of Whispers. Travellers build a scalar autograd engine, then a "
       "tensor one, from nothing but the chain rule. The floor ends where gradients begin to vanish."),
    _p("p06", "The Vanishing Wraith",
       "The Vanishing Wraith haunts Floor 2. It feeds on gradients that shrink with every layer "
       "they pass through until nothing reaches the first weights. Travellers defeat it with "
       "careful initialization and activations that do not saturate."),
    _p("p07", "The Forge of Layers",
       "Floor 3 is the Forge of Layers. Its rooms hammer out multilayer perceptrons, weight "
       "initialization, mini-batch training loops and validation. Regularization is the "
       "tempering that keeps a network from shattering on unseen data."),
    _p("p08", "The Overfit Hydra",
       "The Overfit Hydra grows a new head for every training example it memorizes. Its weakness "
       "is a held-out validation set, which shows that a hundred heads recite the training data "
       "and understand nothing. Regularization and early stopping keep the heads from growing back."),
    _p("p09", "The Torchlit Passage",
       "Floor 4 is the Torchlit Passage, the first floor lit by PyTorch. Travellers meet tensors "
       "with autograd, nn.Module, DataLoader and a gallery of cursed training loops to debug. "
       "Determinism is taught here because nothing below can be debugged without it."),
    _p("p10", "The Reproducibility Revenant",
       "The Reproducibility Revenant returns with a different loss every time it is killed. Its "
       "weakness is a fixed seed for every random generator and deterministic algorithms. Once "
       "the run repeats exactly, the Revenant stays dead."),
    _p("p11", "The Scriptorium of Tokens",
       "Floor 5 is the Scriptorium of Tokens. Scribes there build character and byte-pair "
       "tokenizers, wrestle with Unicode, and learn embeddings, positional encodings and token "
       "budgets. Every word a model reads was first cut into tokens on this floor."),
    _p("p12", "The Babel Golem",
       "The Babel Golem speaks every language at once and understands none. It is defeated with "
       "a byte-level byte-pair encoder whose base vocabulary is all 256 byte values, so any "
       "string that UTF-8 can encode can be tokenized. Decoding half a multi-byte character "
       "yields a replacement character, not a crash."),
    _p("p13", "The Hall of a Thousand Heads",
       "Floor 6 is the Hall of a Thousand Heads. Its rooms teach scaled dot-product attention, "
       "causal and padding masks, and multi-head attention. The last room hunts for leakage."),
    _p("p14", "The Oracle Who Peeks",
       "The Oracle Who Peeks tells the future because it is reading it: a bad causal mask lets "
       "every position attend to the tokens after it. Its weakness is a mask that hides position "
       "j from position i whenever j is greater than i. The trial detects the leak by changing a "
       "future token and watching the past change."),
    _p("p15", "The Tower of the Transformer",
       "Floor 7 is the Tower of the Transformer. Travellers assemble LayerNorm and transformer "
       "blocks into a small GPT and train it on the dungeon's own chronicles. The top of the "
       "tower is about sampling: how a trained model chooses its next token."),
    _p("p16", "The Stuttering Sovereign",
       "The Stuttering Sovereign repeats the same three words forever because it is sampled "
       "greedily. Its weakness is temperature, top-k and nucleus sampling, which let the model "
       "say something new without saying something senseless."),
    _p("p17", "The Vault of Frozen Weights",
       "Floor 8 is the Vault of Frozen Weights. Its rooms cover checkpoints, freezing parameters, "
       "low-rank adapters built from scratch and fine-tuning. The vault is cold because most of "
       "the weights never move again."),
    _p("p18", "The Catastrophic Forgetter",
       "The Catastrophic Forgetter learns a new task by forgetting every old one. Its weakness is "
       "LoRA: a pair of low-rank matrices trained beside frozen weights, so the original knowledge "
       "is never overwritten."),
    _p("p19", "The Library of Echoes",
       "Floor 9 is the Library of Echoes. Every book answers when you call to it, but the echo is "
       "only as good as the question and the shelving. The rooms teach embeddings, cosine "
       "similarity, vector indexes, chunking and retrieval-augmented generation."),
    _p("p20", "The Hallucinating Librarian",
       "The Hallucinating Librarian answers every question confidently and always with a citation, "
       "whether or not the book exists. Its weakness is a reader who refuses when retrieval is weak "
       "and verifies every citation against the shelf before trusting it."),
    _p("p21", "The Court of the Prompt Weaver",
       "Floor 10 is the Court of the Prompt Weaver. Its rooms teach structured outputs, "
       "tool-calling loops, retries with backoff and context budgets. The court's last lesson is "
       "how to defend a model against instructions smuggled in from outside."),
    _p("p22", "The Injected Imp",
       "The Injected Imp hides instructions inside tool results and web pages, hoping the model "
       "will obey them. Its weakness is treating everything a tool returns as data, never as a "
       "command."),
    _p("p23", "The Proving Grounds",
       "Floor 11 is the Proving Grounds. Travellers learn classification metrics, perplexity, "
       "bootstrap confidence intervals and how to use one model to judge another. An eval "
       "harness built here catches regressions before users do."),
    _p("p24", "The Goodhart Gorgon",
       "The Goodhart Gorgon turns any metric you optimise into stone: once the number becomes the "
       "target, it stops measuring what you cared about. Its weakness is a held-out evaluation "
       "with confidence intervals and a judge that is checked against human labels."),
    _p("p25", "The Engine Room",
       "Floor 12 is the Engine Room, the deepest floor. Its rooms cover KV caching, int8 "
       "quantization, batching and latency percentiles, and end with a streaming inference "
       "server. It is loud, hot and measured in milliseconds."),
    _p("p26", "The Latency Leviathan",
       "The Latency Leviathan swallows requests whole and returns them one token at a time, "
       "slowly. Its weakness is a KV cache, which stores the keys and values of past tokens so "
       "each new token costs one step instead of the whole sequence again."),
    # ------------------------------------------------------------- the rules
    _p("p27", "How trials work",
       "Every room is a Python file whose functions raise NotImplementedError until a traveller "
       "writes them. A trial is a pytest file that judges the room. A room counts as cleared "
       "only when its whole trial file passes in one run."),
    _p("p28", "Hints",
       "Each room has exactly three hints. The command dungeon hint reveals one hint per "
       "invocation and records how many were used. The first hint points at the idea, the "
       "second at the tool, and the third is nearly the code."),
    _p("p29", "The save file",
       "Progress is stored in .dungeon/progress.json, which is git-ignored. It is your save "
       "file. Runs in reference-solutions mode never write to it."),
    _p("p30", "Secret rooms",
       "A floor has at most one secret room. Secret rooms are optional and never required to "
       "clear a floor. Clearing one grants bragging rights and nothing else; the floor's loot "
       "unlocks when the boss falls."),
    _p("p31", "Loot",
       "Loot unlocks when every regular room and the boss of a floor are cleared. Loot is "
       "something a traveller reuses afterwards: a cheat sheet, a checklist or a clean reference "
       "implementation."),
    _p("p32", "The Grimoire of Shapes",
       "The Grimoire of Shapes and Broadcasting is the loot of Floor 0. It fits on one page and "
       "lists the four broadcasting rules, the axis-insertion table and the five shape bugs "
       "everyone hits."),
    _p("p33", "The RAG Checklist",
       "The loot of Floor 9 is the RAG Checklist and a clean vector index. The checklist covers "
       "chunking, normalisation, index choice, thresholds, prompt structure and citation "
       "verification, in the order you will need them."),
    _p("p34", "The Whispering Saddle",
       "The Whispering Saddle is the secret room of Floor 1, reached from a side ledge. A saddle "
       "point has zero gradient but is not a minimum. Momentum carries a traveller across it."),
    _p("p35", "The Einsum Oubliette",
       "The Einsum Oubliette is the secret room of Floor 0. Only one incantation is permitted "
       "inside: einsum. Scratched into the wall, over and over, is bij,bjk->bik."),
    _p("p36", "The chronicles",
       "The dungeon's chronicles are the text the tiny GPT of Floor 7 is trained on. They are "
       "stored beside the engine as chronicles.txt, together with the goblin ledger that the "
       "Chronicler uses for its arithmetic."),
    # --------------------------------------------------------- library lore
    _p("p37", "Echoes",
       "An echo in the library is a cosine similarity: the dot product of two unit vectors. Two "
       "books echo each other loudly when their vectors point the same way, and not at all when "
       "they are orthogonal."),
    _p("p38", "The card catalogue",
       "The library's card catalogue is an inverted file: every book is assigned to the shelf "
       "whose centroid is nearest, and a query visits only the few nearest shelves. Visiting more "
       "shelves raises recall and costs more comparisons."),
    _p("p39", "Shelving",
       "Books in the library are shelved in chunks, not whole. A chunk that is too small loses "
       "its context; a chunk that is too large drowns the answer in unrelated sentences. The "
       "scribes overlap consecutive chunks so that no sentence is cut in half at a boundary."),
    _p("p40", "The Librarian's ledger",
       "The Librarian keeps a ledger of every citation it has ever issued. Most entries point to "
       "books that exist. Some point to shelves that were never built."),
    _p("p41", "Refusing",
       "A reader who cannot find an answer in the sources must say so. In the library, the phrase "
       "is: I cannot find that in the library. Refusing is not failure; inventing is."),
    _p("p42", "Grounding",
       "An answer is grounded when every claim in it can be traced to a retrieved source. A "
       "citation is a claim about a source, and claims are verified, not trusted."),
    _p("p43", "Recall",
       "Recall at k is the fraction of the true k nearest neighbours that an approximate search "
       "returns. The card catalogue trades a little recall for a large saving in comparisons; a "
       "flat scan has perfect recall and pays for every book."),
    _p("p44", "Heads, two kinds",
       "The Hall of a Thousand Heads is not a hall of trophies; the heads are attention heads, "
       "each looking at the sequence from a different angle. The Hydra of Floor 3 is a different "
       "beast, and its heads are memorised examples."),
    # ------------------------------------------------------ near-duplicates
    _p("p45", "The Basilisk, again",
       "At the exit of Floor 0 waits the Broadcasting Basilisk. Because its gaze turns numpy's "
       "broadcasting helpers to stone, you must work the four rules on shape tuples yourself. "
       "It is weak against anyone who knows those rules by heart."),
    _p("p46", "On hints",
       "Three hints are written for every room. Running dungeon hint shows one more hint each "
       "time and notes how many you have seen. Hint one is the idea, hint two the tool, hint "
       "three almost the code."),
    _p("p47", "Echoes (a second copy)",
       "An  echo in the library is a cosine similarity:  the dot product of two unit vectors. "
       "Two books echo each other loudly when their vectors point the same way,  and not at all "
       "when they are orthogonal. "),
    # ---------------------------------------------------------------- decoys
    _p("p48", "The refectory's chunks",
       "The refectory serves stew in chunks the size of a fist, with an overlap of gravy where two "
       "ladles were poured touching. Nobody has ever needed a citation to prove the stew was good."),
    _p("p49", "The Librarian's index finger",
       "The Librarian points with its index finger at any shelf you name, whether the shelf is "
       "flat, inverted or imaginary. The finger is always confident; the shelf is not always there."),
    _p("p50", "Late fees",
       "The library charges a late fee that grows at a fixed rate for every day a book is overdue. "
       "The rate does not decay, does not warm up and is not scheduled; it is simply owed."),
    _p("p51", "Embedded gems",
       "The door to the reading room is embedded with forty gems, each a different colour. They "
       "are decorative and have no vector, no norm and no dimension worth speaking of."),
)

GOLDEN_QA: tuple[QA, ...] = (
    QA("What does the Broadcasting Basilisk's gaze do?",
       "petrifies numpy's broadcasting helpers", "p02"),
    QA("What is the weakness of the Learning-Rate Lich?",
       "a learning rate that changes over time", "p04"),
    QA("What does the Overfit Hydra grow for every training example it memorizes?",
       "a new head for every training example it memorizes", "p08"),
    QA("When does the Reproducibility Revenant stay dead?",
       "Once the run repeats exactly, the Revenant stays dead", "p10"),
    QA("What is the base vocabulary of the byte-level encoder that defeats the Babel Golem?",
       "all 256 byte values", "p12"),
    QA("How does the trial detect the Oracle Who Peeks leaking the future?",
       "by changing a future token and watching the past change", "p14"),
    QA("Why does the Stuttering Sovereign repeat the same three words?",
       "because it is sampled greedily", "p16"),
    QA("What does the KV cache that defeats the Latency Leviathan store?",
       "stores the keys and values of past tokens", "p26"),
    QA("When does a room count as cleared?",
       "only when its whole trial file passes in one run", "p27"),
    QA("Where is progress stored?",
       ".dungeon/progress.json", "p29"),
    QA("What incantation is permitted in the Einsum Oubliette?",
       "Only one incantation is permitted inside: einsum", "p35"),
    QA("How does the Injected Imp attack a model?",
       "hides instructions inside tool results and web pages", "p22"),
)

OFF_TOPIC_QUESTIONS: tuple[str, ...] = (
    "How long should I boil an egg for a runny yolk?",
    "Which planet in the solar system has the most moons?",
    "What is the best way to season a cast-iron skillet?",
    "How far away is the Moon from the Earth?",
    "Why does the sky look blue at midday and red at sunset?",
    "How many strings does a violin have?",
    "Which spices go into a classic tomato soup?",
    "What is the tallest mountain on Mars?",
)


_BY_ID = {p.id: p for p in PASSAGES}


def passage_by_id(pid: str) -> Passage:
    """Look a passage up by its id (KeyError if the library does not hold it)."""
    return _BY_ID[pid]


def texts() -> list[str]:
    """Every passage's text, in corpus order."""
    return [p.text for p in PASSAGES]


def ids() -> list[str]:
    """Every passage's id, in corpus order."""
    return [p.id for p in PASSAGES]
