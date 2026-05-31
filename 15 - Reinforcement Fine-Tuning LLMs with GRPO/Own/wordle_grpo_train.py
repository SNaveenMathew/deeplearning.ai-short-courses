#!/usr/bin/env python3
"""
Wordle GRPO Training
====================
Fine-tunes an open-weight LLM to play Wordle using
Group Relative Policy Optimization (GRPO) via TRL + LoRA.

Hardware : NVIDIA RTX 4090 (16 GB VRAM)
Base model: Qwen/Qwen2.5-1.5B-Instruct  (swap to 3B for better quality)
Method   : GRPO + LoRA (parameter-efficient fine-tuning)

How GRPO works here
-------------------
1. A random Wordle game-state (0-4 prior guesses already made) is sampled.
2. The model generates G=8 candidate guesses (group) with temperature sampling.
3. Each guess is scored by a set of reward functions (validity, letter accuracy,
   solve bonus, no-repeated-eliminated-letters penalty).
4. GRPO normalises rewards within the group and updates the policy using a
   clipped policy-gradient objective + KL penalty against the reference model.
5. Over many steps the model learns to produce valid, high-information guesses
   and eventually to solve the puzzle.

Setup
-----
    pip install -r requirements.txt

Train
-----
    python wordle_grpo_train.py
    python wordle_grpo_train.py --model Qwen/Qwen2.5-3B-Instruct  # bigger model
    python wordle_grpo_train.py --steps 1000                       # more training

Play (interactive demo with trained model)
------------------------------------------
    python wordle_grpo_train.py --play
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MAX_GUESSES        = 6
WORD_LENGTH        = 5


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./wordle_grpo_checkpoints"

    min_prior_guesses: int = 2   # ← new: never train on blank-board states
    max_prior_guesses: int = 5   # ← extend to near-endgame
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str  = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str  = "./wordle_grpo_checkpoints"

    # ── Dataset ───────────────────────────────────────────────────────────────
    num_train:        int = 4_000   # training examples (random game-states)
    num_eval:         int = 100     # evaluation examples

    # ── GRPO hyper-parameters ─────────────────────────────────────────────────
    num_generations:  int   = 4     # G — completions per prompt (group size)
    max_new_tokens:   int   = 12   # room for <think> reasoning + 5-letter word
    temperature:      float = 0.9   # generation temperature (diversity)
    beta:             float = 0.04 # KL penalty coefficient (small → explore more)

    # ── Optimisation ──────────────────────────────────────────────────────────
    learning_rate:                float = 3e-6
    per_device_train_batch_size:  int   = 4    # 2 prompts × 8 gens = 16 completions/step
    gradient_accumulation_steps:  int   = 4    # effective batch = 16 prompts
    max_steps:                    int   = 300
    save_steps:                   int   = 100
    eval_steps:                   int   = 100
    logging_steps:                int   = 10
    warmup_steps:                 int   = 30

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r:       int   = 16
    lora_alpha:   int   = 32
    lora_dropout: float = 0.05

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int  = 42
    bf16: bool = True   # RTX 4090 natively supports bfloat16


CFG = TrainConfig()


# ══════════════════════════════════════════════════════════════════════════════
# WORD LIST  (~500 common 5-letter Wordle-style words, embedded for portability)
# ══════════════════════════════════════════════════════════════════════════════
# You can replace / extend this list with the full official Wordle word lists:
#   Answers : https://bit.ly/wordle-answers
#   Guesses : https://bit.ly/wordle-guesses

_RAW_WORDS = """
about above abuse actor acute admit adopt adult after again agent agree ahead
alarm album alert alike align alive alley allow alone along alter angel anger
angle angry ankle annex annoy apart apple apply arena argue arise armor aroma
arose array arrow aside asset atlas atone attic audio audit avail avoid awful
baker basic basis batch beard beast began begin being below bench bible birth
black blade blame bland blank blast blaze bleed blend bless blind block bloom
blown board boast bonus boost booth bound braid brain brand brave brawl breed
bribe brick bride brief bring broad build built burst buyer cabin cache candy
catch cause cease cedar chain chair chant chaos charm chart chase cheap check
cheek cheer chest child chill chord civic civil claim clamp clash clasp class
clean clear click cliff climb cling clock clone close cloth cloud clout clown
coach coast color combo comic coral couch could count court cover crack craft
crane crash cream creed creek crisp cross crowd crown crude crush crust curve
cycle daily dance datum debut decay decoy defer deity delta dense depot depth
devil digit dirty disco ditch dizzy dodge donor doubt dough draft drain drama
drape drawl dream dress drill drink drive drone drool drown drunk dryer dying
eager eagle early earth eerie eight elite ember empty enter entry envoy equal
essay event every evoke exert exile expel extra fable faith false fancy farce
fatal feast fetch fewer fight final first fixed flair flank flare flash flask
fleet flesh fling float flood floor flour fluid focus folly forge forte forum
found frame fraud fresh front frost frown froze fruit fully funny fussy fuzzy
gauge gavel giddy given gland glare glass gleam glide glint gloat gloom glory
gloss glove grace grade grain grant grasp grass grate grave gravy graze great
greet grief grime grind groan groin groom gross group grove growl grown grunt
guard guile guise gusto habit harsh haven heart heavy heist hence heron hoist
holly honor horse hotel hound house human humid humor hurry hyena ideal image
inane inept inert infer inlet inner issue ivory japan jewel joust judge juice
jumpy karma knack knave kneel knife knock known labor lance lanky large latch
later lathe latte laugh layer leach learn lease leash leave ledge legal lemon
level light linen liver local lodge logic loose lover lucid lucky lunar lunch
lyric magic major maker mania manor maple march marry match maxim media merit
merry messy midst might minor minus model moist money month moose moral morph
mossy mount mourn muddy murky music nasal naval nerve never night noble noise
north notch novel nurse ocean offer onset opera optic order other outer oxide
ozone paint panel panic paper party pasta patch pause peace penny perch peril
phase photo piano pilot pixel pizza place plain plane plant plate plaza plead
pluck plumb plume plump plush point poise polar power prank press price pride
prime print prior prism prize probe prone proof prose proud prove prune query
quick quiet quirk quota quite rabid radar raise rally ranch range rapid reach
ready realm reign relax relic repay resin retro revel ridge rifle right risky
roast rocky rouge rough round royal ruddy ruler rural rusty sadly saint salve
sandy sauce savor savvy scald scalp scamp scare scarf scene score scout scrap
scrub seize serve setup seven shade shady shame shape share shark sharp shear
shelf shell shift shirt shock shoot shore short shout shove shred shrug slice
slide slime sling slope sloth small smart smash smear smile smirk smoke smoky
snack snail snake snare snarl sneak sneer spine spite split spoke spook spoon
spore sport spree squad staff stage stain stair stall stamp stand stare stark
start state steam steel steep steer stern stiff still sting stock stoic stomp
stony store storm story stout strap straw strip strut study stuff stump stunt
style suave sugar suite sunny super surge swamp swarm swear sweat sweep sweet
swift swirl swoop syrup taboo taunt tawny teach tense tepid terse testy thank
theft their theme there thick thief thing think thorn those three throb throw
tiger timid tinge tired titan toast tonal tooth total tough toxic trace track
trade trail train trash treat trend trial trick troop trout truce truly trunk
tumor tweak twist tying ulcer ultra uncle under union until upper upset urban
usage usher usual utter veins verge vigor vital vivid vocal voice vomit vouch
vowel wacky waltz watch water weary weave weird where which while whirl white
witch woman world worse worst wound wrath wring wrote yacht yearn yield young
youth zesty
"""

WORDLE_WORDS: list[str] = sorted(
    {w.upper() for w in _RAW_WORDS.split() if len(w) == 5 and w.isalpha()}
)
VALID_WORDS: set[str] = set(WORDLE_WORDS)


# ══════════════════════════════════════════════════════════════════════════════
# WORDLE GAME ENGINE
# ══════════════════════════════════════════════════════════════════════════════

EMOJI = {"G": "🟩", "Y": "🟨", "B": "⬛"}


def get_completion_text(completion) -> str:
    """
    TRL passes completions as either a plain string or a list of message dicts.
    This handles both formats.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # e.g. [{"role": "assistant", "content": "..."}]
        for msg in reversed(completion):
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
    return str(completion)


def get_feedback(guess: str, target: str) -> str:
    """
    Compute letter-by-letter Wordle feedback.

    Returns a 5-char string where each character is:
      G  correct letter, correct position  (green)
      Y  correct letter, wrong position    (yellow)
      B  letter not in the word            (black/gray)
    """
    result    = ["B"] * 5
    remaining = list(target)  # tracks letters still available for yellow

    # Pass 1: greens — exact matches take priority
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            result[i]    = "G"
            remaining[i] = None  # consumed

    # Pass 2: yellows — remaining letters in wrong position
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in remaining:
            result[i] = "Y"
            remaining[remaining.index(g)] = None

    return "".join(result)


def feedback_to_emoji(feedback: str) -> str:
    return "".join(EMOJI[c] for c in feedback)


class WordleGame:
    """Lightweight Wordle game used for simulation and interactive play."""

    def __init__(self, target: str, max_guesses: int = 6):
        self.target     = target.upper()
        self.max_guesses = max_guesses
        self.guesses:   list[str] = []
        self.feedbacks: list[str] = []

    # ── State properties ──────────────────────────────────────────────────────
    @property
    def turn(self) -> int:
        """0-indexed: 0 = no guesses yet."""
        return len(self.guesses)

    @property
    def solved(self) -> bool:
        return bool(self.feedbacks) and self.feedbacks[-1] == "GGGGG"

    @property
    def failed(self) -> bool:
        return not self.solved and self.turn >= self.max_guesses

    @property
    def done(self) -> bool:
        return self.solved or self.failed

    # ── Actions ───────────────────────────────────────────────────────────────
    def step(self, guess: str) -> str:
        """Submit a guess. Returns the feedback string."""
        guess = guess.upper()
        fb    = get_feedback(guess, self.target)
        self.guesses.append(guess)
        self.feedbacks.append(fb)
        return fb


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert Wordle player.

RULES:
• Guess the secret 5-letter English word in at most 6 attempts.
• After each guess you receive feedback for every letter:
    🟩 = correct letter, correct position
    🟨 = correct letter, wrong position
    ⬛ = letter not in the word at all
• Use the feedback to eliminate possibilities and home in on the target.

STRATEGY:
• Start broad: cover common vowels and consonants (e.g. CRANE, STALE, AUDIO).
• Never reuse a letter marked ⬛ — it wastes a guess.
• Place 🟨 letters in a new position on the next guess.
• Maximise information gain before committing to your final answer.

OUTPUT FORMAT — follow this exactly:
<think>
  Reason through the feedback here.
  List eliminated letters, confirmed positions, and remaining candidates.
  Explain why you are choosing your next guess.
</think>
YOURWORD

The LAST LINE must be exactly the 5-letter uppercase word you are guessing.
No punctuation, spaces, or extra text on that final line.
"""


def build_board_text(guesses: list[str], feedbacks: list[str]) -> str:
    """Format the Wordle board for the prompt."""
    if not guesses:
        return "  (no guesses yet — this is your first guess)"
    lines = []
    for g, f in zip(guesses, feedbacks):
        lines.append(f"  {feedback_to_emoji(f)}  {' '.join(g)}")
    return "\n".join(lines)


SYSTEM_PROMPT_TRAIN = """\
You are a Wordle solver. Output ONLY a single 5-letter uppercase English word as your guess. \
No explanation, no punctuation, no other text — just the word on its own line.

Rules:
- Never use a letter already marked as absent (⬛)
- Place 🟨 letters in a new position
- Lock in 🟩 letters at their confirmed position
"""

SYSTEM_PROMPT_PLAY = """\
You are an expert Wordle player.

RULES:
- Guess the secret 5-letter English word in at most 6 attempts.
- After each guess you receive feedback for every letter:
    🟩 = correct letter, correct position
    🟨 = correct letter, wrong position
    ⬛ = letter not in the word at all

OUTPUT FORMAT — follow this exactly:
<think>
  Reason through the feedback. List eliminated letters, confirmed positions,
  and remaining candidates. Explain your choice.
</think>
YOURWORD

The LAST LINE must be exactly the 5-letter uppercase word — no extra text.
"""


def build_prompt_messages(
    guesses:   list[str],
    feedbacks: list[str],
    turn:      int,
    training:  bool = False,    # ← add this
) -> list[dict]:
    remaining = MAX_GUESSES - turn

    eliminated: set[str]       = set()
    confirmed:  dict[int, str] = {}
    in_word:    set[str]       = set()
    for g, f in zip(guesses, feedbacks):
        for i, (c, fb) in enumerate(zip(g, f)):
            if fb == "B":   eliminated.add(c)
            elif fb == "G": confirmed[i] = c
            elif fb == "Y": in_word.add(c)
    in_word -= set(confirmed.values())

    board = (
        "\n".join(
            f"  {feedback_to_emoji(f)}  {' '.join(g)}"
            for g, f in zip(guesses, feedbacks)
        )
        or "  (no guesses yet)"
    )

    user_content = (
        f"Attempt {turn + 1} of {MAX_GUESSES}. Guesses remaining: {remaining}\n"
        f"{board}\n\n"
        f"Eliminated (⬛):              {', '.join(sorted(eliminated)) or 'none'}\n"
        f"Confirmed positions (🟩):    "
        f"{' '.join(f'pos{i+1}={c}' for i, c in sorted(confirmed.items())) or 'none'}\n"
        f"In word, wrong position (🟨): {', '.join(sorted(in_word)) or 'none'}\n\n"
        "Your guess:"
    )

    # Use the no-think prompt during training, full reasoning prompt for play
    system = SYSTEM_PROMPT_TRAIN if training else SYSTEM_PROMPT_PLAY
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# GUESS PARSER
# ══════════════════════════════════════════════════════════════════════════════

def extract_guess(text: str) -> Optional[str]:
    """
    Parse the model's 5-letter word from its completion.

    Strategy (in priority order):
    1. Last non-empty line — expected output position.
    2. Token immediately after </think> tag.
    3. Last 5-letter alphabetic token anywhere in the text.
    """
    text = get_completion_text(text)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # 1. Last line
    if lines:
        candidate = lines[-1].upper()
        if re.fullmatch(r"[A-Z]{5}", candidate):
            return candidate

    # 2. After </think>
    m = re.search(r"</think>\s*([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().upper()
        if len(candidate) == 5:
            return candidate

    # 3. Last 5-letter word in text
    words = re.findall(r"\b[A-Za-z]{5}\b", text)
    if words:
        return words[-1].upper()

    return None


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# TRL's GRPOTrainer calls each reward function with:
#   reward_fn(prompts, completions, **dataset_columns)
#
# When num_generations=G and batch_size=B, there are B*G completions.
# Dataset column tensors/lists are automatically repeated G times by TRL,
# so `target_words[i]` corresponds to `completions[i]` one-to-one.
#
# Return a plain Python list[float] of length len(completions).

def _parse_all(
    completions: list, target_words: list[str]
) -> list[tuple[Optional[str], str, Optional[str]]]:
    results = []
    for comp, target in zip(completions, target_words):
        text  = get_completion_text(comp)   # ← unwrap here
        guess = extract_guess(text)
        if guess is None:
            results.append((None, target.upper(), None))
        else:
            fb = get_feedback(guess, target.upper())
            results.append((guess, target.upper(), fb))
    return results


# ── Reward 1: Format & Validity ───────────────────────────────────────────────
def reward_format_and_validity(
    prompts, completions, *, target_words, **_kw
) -> list[float]:
    rewards = []
    for comp in completions:
        text  = get_completion_text(comp).strip()
        # In no-think mode the model should output just the word
        # Try the last token first, then fall back to extract_guess
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        guess = lines[-1].upper() if lines else None
        if guess and not re.fullmatch(r"[A-Z]{5}", guess):
            guess = extract_guess(text)

        if guess is None:
            rewards.append(-1.0)
        elif guess in VALID_WORDS:
            rewards.append(1.0)   # ← higher reward now that format is simple
        else:
            rewards.append(0.1)
    return rewards


# ── Reward 2: Letter Match Score ──────────────────────────────────────────────
def reward_letter_match(
    prompts: list,
    completions: list[str],
    *,
    target_words: list[str],
    **_kw,
) -> list[float]:
    """
    Score based on how many letters the guess gets right.

    Per-letter rewards:
      🟩 (green)  +0.6
      🟨 (yellow) +0.2
      ⬛ (black)   0.0

    Maximum: 5 × 0.6 = 3.0 (all green, i.e. solved)
    Returns 0.0 for invalid / unparseable guesses.
    """
    rewards = []
    for guess, target, fb in _parse_all(completions, target_words):
        if fb is None:
            rewards.append(0.0)
        else:
            score = fb.count("G") * 0.6 + fb.count("Y") * 0.2
            rewards.append(score)
    return rewards


# ── Reward 3: Solve Bonus ─────────────────────────────────────────────────────
def reward_solve(
    prompts: list,
    completions: list[str],
    *,
    target_words: list[str],
    turn_numbers: list[int],
    **_kw,
) -> list[float]:
    """
    Large bonus when the guess solves the puzzle. Decays with turn number
    to incentivise solving early.

    Turn 1 → +6.0  (lucky guess!)
    Turn 2 → +5.0
    Turn 3 → +4.0
    Turn 4 → +3.0
    Turn 5 → +2.0
    Turn 6 → +1.0
    Not solved → 0.0
    """
    rewards = []
    for (guess, target, fb), turn in zip(
        _parse_all(completions, target_words), turn_numbers
    ):
        if fb == "GGGGG":
            rewards.append(float(7 - (turn + 1)))  # turn is 0-indexed
        else:
            rewards.append(0.0)
    return rewards


# ── Reward 4: No Repeated Eliminated Letters ──────────────────────────────────
def reward_no_repeated_eliminated(
    prompts: list,
    completions: list[str],
    *,
    target_words: list[str],
    eliminated_letters: list[str],   # JSON-encoded list[str] per example
    **_kw,
) -> list[float]:
    """
    Penalises the model for re-using a letter already known to be absent.
    -0.3 per repeated eliminated letter in the guess.

    This teaches the model to respect its own prior feedback.
    """
    rewards = []
    for comp, elim_json in zip(completions, eliminated_letters):
        guess     = extract_guess(comp)
        eliminated = set(json.loads(elim_json))
        if guess is None or not eliminated:
            rewards.append(0.0)
            continue
        n_bad = sum(1 for c in guess if c in eliminated)
        rewards.append(-0.3 * n_bad)
    return rewards


# ── Reward 5: Information Diversity ───────────────────────────────────────────
def reward_new_information(
    prompts, completions, *, target_words, eliminated_letters, confirmed_letters, **_kw
) -> list[float]:
    rewards = []
    for comp, elim_json, conf_json in zip(completions, eliminated_letters, confirmed_letters):
        guess = extract_guess(get_completion_text(comp))   # ← unwrap here
        known = set(json.loads(elim_json)) | set(json.loads(conf_json))
        if guess is None:
            rewards.append(0.0)
            continue
        novel = len({c for c in set(guess) if c not in known})
        rewards.append(0.1 * novel)
    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_prior_guesses(
    target: str, n: int, word_pool: list[str]
) -> tuple[list[str], list[str]]:
    """
    Randomly simulate `n` guesses before the target turn using words from
    the pool that are NOT the answer (to avoid accidental early solves).
    """
    pool = [w for w in word_pool if w != target]
    random.shuffle(pool)
    guesses, feedbacks = [], []
    for word in pool[:n]:
        fb = get_feedback(word, target)
        guesses.append(word)
        feedbacks.append(fb)
        if fb == "GGGGG":
            break  # stop if accidentally solved
    return guesses, feedbacks


def make_example(target: str, min_prior: int, max_prior: int, word_pool: list[str]) -> dict:
    n_prior = random.randint(min_prior, max_prior)
    guesses, feedbacks = simulate_prior_guesses(target, n_prior, word_pool)
    turn = len(guesses)

    eliminated:       set[str] = set()
    confirmed_placed: set[str] = set()
    for g, f in zip(guesses, feedbacks):
        for c, fb in zip(g, f):
            if fb == "B":   eliminated.add(c)
            elif fb == "G": confirmed_placed.add(c)

    messages = build_prompt_messages(guesses, feedbacks, turn, training=True)

    return {
        "prompt":             messages,
        "target_words":       target,
        "turn_numbers":       turn,
        "eliminated_letters": json.dumps(list(eliminated)),
        "confirmed_letters":  json.dumps(list(confirmed_placed)),
    }


def build_dataset(cfg: TrainConfig, word_list: list[str]) -> tuple[Dataset, Dataset]:
    random.seed(cfg.seed)

    def gen(n: int) -> list[dict]:
        return [
            make_example(
                random.choice(word_list),
                cfg.min_prior_guesses,   # ← min
                cfg.max_prior_guesses,   # ← max
                word_list,
            )
            for _ in range(n)
        ]

    return Dataset.from_list(gen(cfg.num_train)), Dataset.from_list(gen(cfg.num_eval))


def make_example_curriculum(target: str, word_pool: list[str]) -> dict:
    """
    Sample prior guesses with a skewed distribution:
    favour constrained states where the model can actually learn to solve.

    Probability by n_prior:
      0 prior:  5%   (blank board — mostly noise)
      1 prior: 10%
      2 prior: 20%
      3 prior: 30%   ← model sees ~10-50 candidates, high reward variance
      4 prior: 25%
      5 prior: 10%   ← often 1-3 candidates remain — easy solve signal
    """
    n_prior = random.choices(
        population=[0, 1, 2, 3, 4, 5],
        weights   =[5, 10, 20, 30, 25, 10],
    )[0]
    return make_example(target, n_prior, n_prior, word_pool)


def reward_solve(prompts, completions, *, target_words, turn_numbers, **_kw) -> list[float]:
    rewards = []
    for (guess, target, fb), turn in zip(
        _parse_all(completions, target_words), turn_numbers
    ):
        if fb == "GGGGG":
            # Large flat bonus — model should always try to solve
            rewards.append(10.0)
        else:
            rewards.append(0.0)
    return rewards


def reward_candidate_rank(
    prompts, completions, *, target_words, eliminated_letters, confirmed_letters, **_kw
) -> list[float]:
    """
    Reward the model for choosing a word that is still consistent with
    the observed feedback (i.e. in the candidate pool).
    +1.0 if the guess is consistent with all prior feedback
     0.0 if it contradicts known constraints
    This provides fine-grained signal even when other rewards are identical.
    """
    rewards = []
    for comp, elim_json, conf_json in zip(completions, eliminated_letters, confirmed_letters):
        guess     = extract_guess(get_completion_text(comp))
        elim      = set(json.loads(elim_json))
        confirmed = json.loads(conf_json)  # list of (pos, letter) or just letters

        if guess is None:
            rewards.append(-0.5)
            continue

        # Simple consistency check: no eliminated letters used
        uses_eliminated = any(c in elim for c in guess)
        rewards.append(0.0 if uses_eliminated else 1.0)

    return rewards


# ══════════════════════════════════════════════════════════════════════════════
# MODEL & TRAINER SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(cfg: TrainConfig):
    """Load base model in bfloat16 onto the GPU."""
    print(f"⏳  Loading {cfg.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    # Some models (e.g. LLaMA) have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="cuda",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False   # required for gradient checkpointing
    return model, tokenizer


def build_lora_config(cfg: TrainConfig) -> LoraConfig:
    """
    LoRA targets the attention projections and MLP gates — all linear layers
    that benefit most from task-specific adaptation.
    Adjust target_modules if you switch to a different architecture.
    """
    return LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = cfg.lora_r,
        lora_alpha   = cfg.lora_alpha,
        lora_dropout = cfg.lora_dropout,
        target_modules = [
            # Attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # MLP (Qwen / LLaMA architecture)
            "gate_proj", "up_proj", "down_proj",
        ],
        bias = "none",
    )


def build_grpo_config(cfg: TrainConfig) -> GRPOConfig:
    return GRPOConfig(
        # ── Paths & logging ──────────────────────────────────────────────────
        output_dir    = cfg.output_dir,
        report_to     = "none",
        logging_steps = cfg.logging_steps,

        # ── Optimisation ─────────────────────────────────────────────────────
        learning_rate                = cfg.learning_rate,
        per_device_train_batch_size  = cfg.per_device_train_batch_size,
        gradient_accumulation_steps  = cfg.gradient_accumulation_steps,
        max_steps                    = cfg.max_steps,
        warmup_steps                 = cfg.warmup_steps,
        bf16                         = cfg.bf16,
        seed                         = cfg.seed,

        # ── Checkpointing & evaluation ───────────────────────────────────────
        save_steps    = cfg.save_steps,
        eval_steps    = cfg.eval_steps,
        eval_strategy = "steps",

        # ── GRPO-specific ─────────────────────────────────────────────────────
        num_generations      = cfg.num_generations,
        max_completion_length = cfg.max_new_tokens,   # ← renamed in your TRL version
        beta                 = cfg.beta,

        # ── Generation kwargs (temperature, etc.) ────────────────────────────
        generation_kwargs = {
            "temperature": cfg.temperature,
            "do_sample":   True,
        },

        # ── Memory / throughput ───────────────────────────────────────────────
        gradient_checkpointing = True,
        dataloader_num_workers = 0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (full-game win-rate benchmark)
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def evaluate_model(
    model,
    tokenizer,
    word_list: list[str],
    n_games:     int  = 100,
    max_guesses: int  = 6,
    verbose:     bool = False,
) -> dict:
    model.eval()
    wins, turns_when_solved = 0, []

    for game_i in range(n_games):
        target = random.choice(word_list)
        game   = WordleGame(target, max_guesses)

        while not game.done:
            messages = build_prompt_messages(
                game.guesses, game.feedbacks, game.turn
            )

            # ── FIX: apply_chat_template may return BatchEncoding or raw tensor ──
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # Handle both: raw tensor and BatchEncoding dict
            if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
                input_ids = tokenized["input_ids"].to(model.device)
            else:
                input_ids = tokenized.to(model.device)

            # output_ids = model.generate(
            #     input_ids,
            #     max_new_tokens     = CFG.max_new_tokens,
            #     temperature        = 0.3,
            #     do_sample          = True,
            #     pad_token_id       = tokenizer.eos_token_id,
            # )
            output_ids = model.generate(
                input_ids,
                max_new_tokens = 20,    # ← was CFG.max_new_tokens (48 or 8), now explicit
                do_sample      = False, # greedy for eval
                pad_token_id   = tokenizer.eos_token_id,
            )
            new_tokens = output_ids[0, input_ids.shape[-1]:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

            guess = extract_guess(completion)
            if verbose:
                print(f"  [game {game_i}] target={target} "
                      f"turn={game.turn} guess={guess}")

            if guess is None or guess not in VALID_WORDS:
                guess = random.choice(word_list)

            game.step(guess)

        if game.solved:
            wins += 1
            turns_when_solved.append(game.turn)

    win_rate  = wins / n_games
    avg_turns = (sum(turns_when_solved) / len(turns_when_solved)
                 if turns_when_solved else float("nan"))

    print(
        f"\n📊  Eval ({n_games} games): "
        f"win rate = {win_rate:.1%}  |  "
        f"avg turns (wins) = {avg_turns:.2f}"
    )
    return {"win_rate": win_rate, "avg_turns_when_won": avg_turns, "n_games": n_games}


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLAY  (watch the trained model play a game)
# ══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def play_interactive(model, tokenizer, reveal_target: bool = False):
    """Watch the model play one Wordle game. Optionally reveal the target."""
    target = random.choice(WORDLE_WORDS)
    game   = WordleGame(target)

    if reveal_target:
        print(f"\n🎯  Secret word: {target}\n")
    else:
        print("\n🎯  The model is playing Wordle (secret word hidden)\n")

    while not game.done:
        messages  = build_prompt_messages(game.guesses, game.feedbacks, game.turn)
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens  = CFG.max_new_tokens,
            temperature     = 0.4,
            do_sample       = True,
            pad_token_id    = tokenizer.eos_token_id,
        )
        new_tokens = output_ids[0, input_ids.shape[-1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        guess = extract_guess(completion) or random.choice(WORDLE_WORDS)
        fb    = game.step(guess)

        # ── Pretty-print ──────────────────────────────────────────────────────
        print(f"  Guess {game.turn}: {guess}  {feedback_to_emoji(fb)}")
        # Show the model's reasoning (first 300 chars)
        think_m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if think_m:
            thought = think_m.group(1).strip()[:300]
            print(f"  💭  {thought}\n")

    print()
    if game.solved:
        print(f"✅  Solved in {game.turn} guess(es)! Word was: {target}")
    else:
        print(f"❌  Failed to solve in {game.max_guesses} guesses. Word was: {target}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning to teach an LLM to play Wordle"
    )
    parser.add_argument("--model",      default=CFG.model_name,
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--output_dir", default=CFG.output_dir)
    parser.add_argument("--steps",      type=int, default=CFG.max_steps,
                        help="Number of training steps")
    parser.add_argument("--play",       action="store_true",
                        help="Skip training; load saved LoRA and watch the model play")
    parser.add_argument("--reveal",     action="store_true",
                        help="(with --play) reveal the secret word before guessing")
    parser.add_argument("--eval_only",  action="store_true",
                        help="Skip training; run full-game win-rate evaluation")
    args = parser.parse_args()

    # Apply CLI overrides to global config
    CFG.model_name = args.model
    CFG.output_dir = args.output_dir
    CFG.max_steps  = args.steps

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(CFG)

    # ── Interactive play mode ─────────────────────────────────────────────────
    if args.play or args.eval_only:
        ckpt = os.path.join(CFG.output_dir, "final")
        if os.path.isdir(ckpt):
            print(f"🔄  Loading LoRA weights from {ckpt} …")
            model = PeftModel.from_pretrained(model, ckpt)
            model = model.merge_and_unload()
        else:
            print("⚠️   No saved checkpoint found — using base model (untrained).")

        if args.play:
            play_interactive(model, tokenizer, reveal_target=args.reveal)
        else:
            evaluate_model(model, tokenizer, WORDLE_WORDS, n_games=200, verbose=False)
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    print(f"📚  Word list: {len(WORDLE_WORDS)} words")
    train_ds, eval_ds = build_dataset(CFG, WORDLE_WORDS)
    print(f"📦  Dataset: {len(train_ds)} train  |  {len(eval_ds)} eval examples")

    grpo_config = build_grpo_config(CFG)
    lora_config = build_lora_config(CFG)

    trainer = GRPOTrainer(
        model         = model,
        args          = grpo_config,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
        reward_funcs = [
            reward_format_and_validity,
            reward_letter_match,
            reward_solve,                  # boosted to +10
            reward_no_repeated_eliminated,
            reward_new_information,
            reward_candidate_rank,         # ← new
        ],
        peft_config   = lora_config,
    )

    print(f"\n🚀  Starting GRPO training for {CFG.max_steps} steps …\n")
    print(f"    Model    : {CFG.model_name}")
    print(f"    Group G  : {CFG.num_generations}  (completions per prompt)")
    print(f"    Batch    : {CFG.per_device_train_batch_size} × {CFG.gradient_accumulation_steps} accum")
    print(f"    LR       : {CFG.learning_rate}")
    print(f"    Output   : {CFG.output_dir}\n")

    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    final_dir = os.path.join(CFG.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\n✅  LoRA adapter saved to {final_dir}")

    # ── Post-training evaluation ───────────────────────────────────────────────
    print("\n📊  Running post-training full-game evaluation …")
    evaluate_model(model, tokenizer, WORDLE_WORDS, n_games=200)


if __name__ == "__main__":
    main()
