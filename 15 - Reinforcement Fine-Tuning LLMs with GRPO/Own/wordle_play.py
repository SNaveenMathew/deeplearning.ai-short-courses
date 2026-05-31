#!/usr/bin/env python3
"""
wordle_play.py — Play one Wordle game with live candidate analysis
==================================================================
Loads your GRPO-trained model (or the base model if no checkpoint
exists yet) and plays one full game of Wordle, showing at each step:

  • The Wordle board so far
  • Every word still consistent with the feedback (constraint-filtered)
  • Each candidate's MODEL SCORE  — log-probability under the LLM
  • Each candidate's INFO GAIN    — expected bits of information
                                    (how much it narrows the pool)
  • The model's <think> reasoning before its final guess

Usage
-----
    python wordle_play.py                         # random target, trained model
    python wordle_play.py --target CRANE          # specific target word
    python wordle_play.py --top 20                # show top-20 candidates
    python wordle_play.py --checkpoint ./my_ckpt  # custom checkpoint path
    python wordle_play.py --base-only             # skip LoRA, use raw base model
    python wordle_play.py --model Qwen/Qwen2.5-3B-Instruct  # different base model

Requirements
------------
    pip install rich   (for pretty tables — falls back to plain text if absent)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Try importing Rich for pretty terminal output ─────────────────────────────
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.rule import Rule
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None
    print("[info] pip install rich for prettier output\n")

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULTS  (mirror TrainConfig so the two scripts stay in sync)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./wordle_grpo_checkpoints/final"
DEFAULT_TOP_N      = 15          # candidates to display per turn
SCORE_BATCH_SIZE   = 32          # words scored per GPU forward-pass batch
from wordle_grpo_train import MAX_GUESSES
# from wordle_grpo_train import WORD_LENGTH

# ══════════════════════════════════════════════════════════════════════════════
# WORD LIST  (same as training script — keep in sync)
# ══════════════════════════════════════════════════════════════════════════════

_RAW_WORDS = """
about above abuse actor acute admit adopt adult after again agent agree ahead
alarm album alert alike align alive alley allow alone along alter angel anger
angle angry ankle annex annoy apart apple apply arena argue arise armor aroma
arose array arrow aside asset atlas atone attic audio audit avail avoid awful
baker basic basis batch beard beard beast began begin being below bench bible birth
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
# WORDLE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

EMOJI = {"G": "🟩", "Y": "🟨", "B": "⬛"}
COLORS = {"G": "green", "Y": "yellow", "B": "white"}


def get_feedback(guess: str, target: str) -> str:
    result    = ["B"] * 5
    remaining = list(target)
    for i, (g, t) in enumerate(zip(guess, target)):
        if g == t:
            result[i]    = "G"
            remaining[i] = None
    for i, g in enumerate(guess):
        if result[i] == "G":
            continue
        if g in remaining:
            result[i] = "Y"
            remaining[remaining.index(g)] = None
    return "".join(result)


def feedback_to_emoji(fb: str) -> str:
    return "".join(EMOJI[c] for c in fb)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT FILTERING
# Keeps only words that are fully consistent with all prior guess+feedback pairs
# ══════════════════════════════════════════════════════════════════════════════

def filter_candidates(
    word_list: list[str],
    guesses:   list[str],
    feedbacks: list[str],
) -> list[str]:
    """
    Hard-constraint filter: keep every word in word_list that is consistent
    with ALL (guess, feedback) pairs seen so far.

    Consistency rules derived from Wordle semantics:
      G at pos i  →  candidate[i] == letter
      Y at pos i  →  letter in candidate  AND  candidate[i] != letter
      B for letter →  letter count in candidate ≤ number of G/Y for that letter
    """
    if not guesses:
        return list(word_list)

    # Pre-compute constraints from all guesses
    required_positions: dict[int, str]  = {}    # pos → must-be letter (green)
    wrong_positions:    dict[str, set[int]] = {} # letter → positions where it CANNOT go (yellow)
    min_counts:         dict[str, int]  = {}    # letter → minimum occurrences in word
    max_counts:         dict[str, int]  = {}    # letter → maximum occurrences (from grey)

    for guess, fb in zip(guesses, feedbacks):
        letter_fb: dict[str, list[str]] = {}
        for c, f in zip(guess, fb):
            letter_fb.setdefault(c, []).append(f)

        for i, (c, f) in enumerate(zip(guess, fb)):
            if f == "G":
                required_positions[i] = c
            elif f == "Y":
                wrong_positions.setdefault(c, set()).add(i)

        for c, fbs in letter_fb.items():
            greens_yellows = sum(1 for f in fbs if f in ("G", "Y"))
            greys          = sum(1 for f in fbs if f == "B")
            min_counts[c]  = max(min_counts.get(c, 0), greens_yellows)
            if greys > 0:
                # At least one grey → exact cap is greens+yellows count
                max_counts[c] = greens_yellows

    candidates = []
    for word in word_list:
        ok = True

        # Check green positions
        for pos, letter in required_positions.items():
            if word[pos] != letter:
                ok = False
                break
        if not ok:
            continue

        # Check yellow positions (letter exists, just not there)
        for letter, bad_positions in wrong_positions.items():
            if letter not in word:
                ok = False
                break
            for pos in bad_positions:
                if word[pos] == letter:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Check letter count constraints
        word_counts = Counter(word)
        for letter, mn in min_counts.items():
            if word_counts[letter] < mn:
                ok = False
                break
        if not ok:
            continue
        for letter, mx in max_counts.items():
            if word_counts[letter] > mx:
                ok = False
                break
        if not ok:
            continue

        candidates.append(word)

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# INFORMATION GAIN  (game-theoretic score independent of the model)
# ══════════════════════════════════════════════════════════════════════════════

def information_gain(guess: str, candidates: list[str]) -> float:
    """
    Expected bits of information gained by guessing `guess` given `candidates`.

    For each possible target in candidates, we compute the feedback pattern.
    The entropy H of that pattern distribution tells us how well this guess
    splits the remaining pool.

      H = -Σ p(pattern) · log2(p(pattern))

    A guess with H = log2(|candidates|) perfectly splits the pool (best case).
    H = 0 means all targets give the same pattern (useless guess).
    """
    if not candidates:
        return 0.0
    pattern_counts: Counter = Counter()
    for target in candidates:
        pattern_counts[get_feedback(guess, target)] += 1
    total = len(candidates)
    return -sum((c / total) * math.log2(c / total) for c in pattern_counts.values())


def expected_remaining(guess: str, candidates: list[str]) -> float:
    """
    Expected number of candidates remaining after guessing `guess`.
    Lower is better.
    """
    if not candidates:
        return 0.0
    pattern_groups: dict[str, int] = {}
    for target in candidates:
        fb = get_feedback(guess, target)
        pattern_groups[fb] = pattern_groups.get(fb, 0) + 1
    total = len(candidates)
    return sum((cnt / total) * cnt for cnt in pattern_groups.values())


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SCORING
# Score each candidate word by its log-probability under the trained LLM
# ══════════════════════════════════════════════════════════════════════════════

def score_candidates_with_model(
    model,
    tokenizer,
    candidates: list[str],
    messages:   list[dict],
    batch_size: int = SCORE_BATCH_SIZE,
) -> dict[str, float]:
    """
    For each candidate word, compute log P(word | prompt) under the LLM.

    We run batched forward passes:
      input  = [prompt tokens] + [word tokens]
      score  = sum of log-softmax values at each word-token position

    Returns a dict {WORD: log_prob_score}.
    Scores are negative (log-probs); higher (less negative) = more likely.
    """
    # Format the prompt once
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # prompt_text = tokenizer.apply_chat_template(
    #     messages, tokenize=False, return_tensors="pt", add_generation_prompt=True
    # )
    prompt_ids: list[int] = tokenizer.encode(
        prompt_text, add_special_tokens=False
    )

    # Tokenise every candidate word
    word_token_map: dict[str, list[int]] = {
        word: tokenizer.encode(word, add_special_tokens=False)
        for word in candidates
    }

    scores: dict[str, float] = {}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    for i in range(0, len(candidates), batch_size):
        batch_words = candidates[i : i + batch_size]

        # Build padded sequences: prompt + word tokens
        sequences = [prompt_ids + word_token_map[w] for w in batch_words]
        max_len   = max(len(s) for s in sequences)

        padded_batch  = []
        attention_batch = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded_batch.append([pad_id] * pad_len + seq)
            attention_batch.append([0] * pad_len + [1] * len(seq))

        input_ids      = torch.tensor(padded_batch,   device=model.device)
        attention_mask = torch.tensor(attention_batch, device=model.device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits  # [B, seq_len, vocab]

        log_probs = torch.log_softmax(logits, dim=-1)  # [B, seq_len, vocab]

        for j, word in enumerate(batch_words):
            word_toks = word_token_map[word]
            seq_len   = len(prompt_ids) + len(word_toks)
            pad_len   = max_len - seq_len

            score = 0.0
            for k, tok_id in enumerate(word_toks):
                # The position that predicts word_toks[k] is:
                #   pad_len + len(prompt_ids) + k - 1   (0-indexed in padded seq)
                pred_pos = pad_len + len(prompt_ids) + k - 1
                score   += log_probs[j, pred_pos, tok_id].item()

            scores[word] = score

    return scores


def softmax_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    """Convert log-prob scores to a normalised probability distribution (%)."""
    if not raw_scores:
        return {}
    words  = list(raw_scores)
    logits = torch.tensor([raw_scores[w] for w in words], dtype=torch.float32)
    probs  = torch.softmax(logits, dim=0).tolist()
    return {w: p * 100.0 for w, p in zip(words, probs)}


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER  (same as training script)
# ══════════════════════════════════════════════════════════════════════════════

# Used ONLY during training — no chain-of-thought, just the word
SYSTEM_PROMPT_TRAIN = """\
You are a Wordle solver. Output ONLY a single 5-letter uppercase English word as your guess. \
No explanation, no punctuation, no other text — just the word on its own line.

Rules:
- Never use a letter already marked as absent (⬛)
- Place 🟨 letters in a new position
- Lock in 🟩 letters at their confirmed position
"""

# Used during play/inference — full reasoning enabled
SYSTEM_PROMPT_PLAY = """\
You are an expert Wordle player.
...  (keep the existing SYSTEM_PROMPT content here)
"""


def build_prompt_messages(
    guesses: list[str],
    feedbacks: list[str],
    turn: int,
    training: bool = False,      # ← new flag
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
        "\n".join(f"  {feedback_to_emoji(f)}  {' '.join(g)}" for g, f in zip(guesses, feedbacks))
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

    system = SYSTEM_PROMPT_TRAIN if training else SYSTEM_PROMPT_PLAY
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, max_val: float, width: int = 12) -> str:
    """ASCII progress bar."""
    filled = int(round(width * value / max_val)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def display_board_rich(guesses: list[str], feedbacks: list[str], target_hidden: bool):
    """Print the Wordle board using Rich."""
    title = "🟩 WORDLE" + (" — target hidden" if target_hidden else "")
    rows  = []
    for g, f in zip(guesses, feedbacks):
        parts = []
        for c, fb in zip(g, f):
            parts.append(f"[bold {COLORS[fb]}]{c}[/]")
        rows.append("  " + "  ".join(parts) + f"   {feedback_to_emoji(f)}")
    body = "\n".join(rows) if rows else "  (board empty — make your first guess)"
    console.print(Panel(body, title=title, border_style="cyan"))


def display_candidates_rich(
    candidates:    list[str],
    model_probs:   dict[str, float],
    info_gains:    dict[str, float],
    exp_remaining: dict[str, float],
    top_n:         int,
    total_pool:    int,
    model_pick:    Optional[str],
):
    """Print a Rich table of the top-N candidate words with scores."""
    max_ig = max(info_gains.values()) if info_gains else 1.0

    # Sort by info gain descending (then model prob as tiebreak)
    ranked = sorted(
        candidates,
        key=lambda w: (info_gains.get(w, 0.0), model_probs.get(w, 0.0)),
        reverse=True,
    )
    top = ranked[:top_n]

    table = Table(
        title=f"Candidates: {len(candidates)} / {total_pool} words remain",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#",          style="dim",    width=3,  justify="right")
    table.add_column("Word",       style="bold",   width=7)
    table.add_column("Model %",    width=9,  justify="right")
    table.add_column("Model bar",  width=14)
    table.add_column("Info (bits)", width=11, justify="right")
    table.add_column("Info bar",   width=14)
    table.add_column("Avg left",   width=9,  justify="right")
    table.add_column("★ pick",     width=7,  justify="center")

    max_prob = max(model_probs.values()) if model_probs else 1.0

    for rank, word in enumerate(top, 1):
        prob  = model_probs.get(word, 0.0)
        ig    = info_gains.get(word, 0.0)
        er    = exp_remaining.get(word, 0.0)
        star  = "⭐" if word == model_pick else ""

        # Colour the word itself
        if word == model_pick:
            word_str = f"[bold green]{word}[/]"
        else:
            word_str = word

        table.add_row(
            str(rank),
            word_str,
            f"{prob:6.2f}%",
            f"[cyan]{_bar(prob, max_prob)}[/]",
            f"{ig:5.3f}",
            f"[yellow]{_bar(ig, max_ig)}[/]",
            f"{er:5.1f}",
            star,
        )

    console.print(table)

    if len(candidates) > top_n:
        console.print(
            f"  [dim]… and {len(candidates) - top_n} more consistent words not shown[/]"
        )


def display_plain(
    guesses, feedbacks, candidates, model_probs, info_gains, exp_remaining,
    top_n, total_pool, model_pick,
):
    """Plain-text fallback when Rich is not installed."""
    print("\n" + "═" * 60)
    print("WORDLE BOARD")
    print("═" * 60)
    for g, f in zip(guesses, feedbacks):
        print(f"  {feedback_to_emoji(f)}  {' '.join(g)}")
    if not guesses:
        print("  (board empty)")

    print(f"\nCandidates: {len(candidates)} / {total_pool} words remain")
    print(f"{'#':>3}  {'WORD':<7}  {'Model%':>7}  {'InfoBits':>9}  {'AvgLeft':>8}  Pick")
    print("-" * 60)

    ranked = sorted(
        candidates,
        key=lambda w: (info_gains.get(w, 0.0), model_probs.get(w, 0.0)),
        reverse=True,
    )
    max_prob = max(model_probs.values()) if model_probs else 1.0
    for rank, word in enumerate(ranked[:top_n], 1):
        prob = model_probs.get(word, 0.0)
        ig   = info_gains.get(word, 0.0)
        er   = exp_remaining.get(word, 0.0)
        pick = "<-- model" if word == model_pick else ""
        print(f"  {rank:>2}  {word:<7}  {prob:6.2f}%  {ig:8.3f}  {er:7.1f}  {pick}")

    if len(candidates) > top_n:
        print(f"  … and {len(candidates) - top_n} more consistent words not shown")


# ══════════════════════════════════════════════════════════════════════════════
# GUESS EXTRACTION  (handles both str and list-of-dicts completions)
# ══════════════════════════════════════════════════════════════════════════════

def get_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]
    return str(completion)


def extract_guess(text) -> Optional[str]:
    text  = get_completion_text(text)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        candidate = lines[-1].upper()
        if re.fullmatch(r"[A-Z]{5}", candidate):
            return candidate
    m = re.search(r"</think>\s*([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().upper()
        if len(candidate) == 5:
            return candidate
    words = re.findall(r"\b[A-Za-z]{5}\b", text)
    if words:
        return words[-1].upper()
    return None


def extract_think(text: str) -> Optional[str]:
    """Pull out the model's <think>...</think> reasoning block."""
    text = get_completion_text(text)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GAME LOOP
# ══════════════════════════════════════════════════════════════════════════════

def play_game(
    model,
    tokenizer,
    target:     str,
    top_n:      int     = DEFAULT_TOP_N,
    reveal:     bool    = False,
    gen_temp:   float   = 0.4,
):
    """
    Play one complete Wordle game, printing full candidate analysis each turn.
    """
    target    = target.upper()
    guesses:  list[str] = []
    feedbacks:list[str] = []
    all_words = WORDLE_WORDS

    if RICH:
        console.print(Rule("[bold cyan]WORDLE  ·  GRPO Model Analysis[/]", style="cyan"))
        if reveal:
            console.print(f"[bold red]🎯 Secret word: {target}[/]\n")
        else:
            console.print("[dim]Secret word is hidden. Use --reveal to show it.[/]\n")
    else:
        print("\n" + "═" * 60)
        print("WORDLE — GRPO Model Analysis")
        if reveal:
            print(f"Secret word: {target}")
        print("═" * 60)

    for turn in range(MAX_GUESSES):

        # ── 1. Filter candidates ───────────────────────────────────────────────
        candidates = filter_candidates(all_words, guesses, feedbacks)

        # ── 2. Score candidates ────────────────────────────────────────────────
        messages = build_prompt_messages(guesses, feedbacks, turn)

        if RICH:
            with console.status("[bold green]Scoring candidates with model…[/]"):
                model_raw   = score_candidates_with_model(model, tokenizer, candidates, messages)
        else:
            print("\nScoring candidates…")
            model_raw = score_candidates_with_model(model, tokenizer, candidates, messages)

        model_probs   = softmax_scores(model_raw)

        if RICH:
            with console.status("[bold yellow]Computing information gain…[/]"):
                info_gains    = {w: information_gain(w, candidates) for w in candidates}
                exp_remaining = {w: expected_remaining(w, candidates) for w in candidates}
        else:
            info_gains    = {w: information_gain(w, candidates) for w in candidates}
            exp_remaining = {w: expected_remaining(w, candidates) for w in candidates}

        # ── 3. Generate the model's actual guess (with reasoning) ──────────────
        if RICH:
            with console.status("[bold cyan]Model is thinking…[/]"):
                tokenized = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
                    input_ids = tokenized["input_ids"].to(model.device)
                else:
                    input_ids = tokenized.to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens  = 256,
                        temperature     = gen_temp,
                        do_sample       = True,
                        pad_token_id    = tokenizer.eos_token_id,
                    )
        else:
            print("Model generating guess…")
            tokenized = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(tokenized, dict) or hasattr(tokenized, "input_ids"):
                input_ids = tokenized["input_ids"].to(model.device)
            else:
                input_ids = tokenized.to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens = 256,
                    temperature    = gen_temp,
                    do_sample      = True,
                    pad_token_id   = tokenizer.eos_token_id,
                )

        new_tokens  = output_ids[0, input_ids.shape[-1]:]
        completion  = tokenizer.decode(new_tokens, skip_special_tokens=True)
        model_guess = extract_guess(completion)
        thought     = extract_think(completion)

        # Fallback: if model output is invalid, use highest info-gain candidate
        if model_guess is None or model_guess not in VALID_WORDS:
            model_guess = max(candidates, key=lambda w: info_gains.get(w, 0.0)) if candidates else random.choice(all_words)
            if RICH:
                console.print(f"[yellow]⚠ Model output invalid — falling back to top info-gain word: {model_guess}[/]")
            else:
                print(f"⚠ Model output invalid — falling back to: {model_guess}")

        # ── 4. Display board + candidates ─────────────────────────────────────
        if RICH:
            console.print()
            display_board_rich(guesses, feedbacks, target_hidden=not reveal)
            console.print()
            display_candidates_rich(
                candidates, model_probs, info_gains, exp_remaining,
                top_n=top_n, total_pool=len(all_words), model_pick=model_guess,
            )
        else:
            display_plain(
                guesses, feedbacks, candidates, model_probs, info_gains,
                exp_remaining, top_n, len(all_words), model_guess,
            )

        # ── 5. Show model reasoning ────────────────────────────────────────────
        if thought:
            if RICH:
                console.print(
                    Panel(
                        thought[:500] + ("…" if len(thought) > 500 else ""),
                        title="[bold]💭 Model Reasoning[/]",
                        border_style="dim yellow",
                    )
                )
            else:
                print(f"\n💭 Reasoning: {thought[:400]}")

        # ── 6. Apply guess ────────────────────────────────────────────────────
        fb = get_feedback(model_guess, target)
        guesses.append(model_guess)
        feedbacks.append(fb)

        guess_rank = sorted(
            candidates,
            key=lambda w: info_gains.get(w, 0.0),
            reverse=True,
        ).index(model_guess) + 1 if model_guess in candidates else "?"

        if RICH:
            console.print(
                f"\n  [bold]Guess {turn + 1}:[/]  "
                f"[bold cyan]{model_guess}[/]   "
                f"{feedback_to_emoji(fb)}   "
                f"[dim](ranked #{guess_rank} by info gain among {len(candidates)} candidates)[/]"
            )
        else:
            print(f"\n  Guess {turn + 1}: {model_guess}  {feedback_to_emoji(fb)}  "
                  f"(ranked #{guess_rank} by info gain among {len(candidates)} candidates)")

        # ── 7. Check win/loss ─────────────────────────────────────────────────
        if fb == "GGGGG":
            if RICH:
                console.print()
                display_board_rich(guesses, feedbacks, target_hidden=False)
                console.print(
                    f"\n[bold green]✅  Solved in {turn + 1} guess(es)! "
                    f"Word was: [underline]{target}[/][/]"
                )
            else:
                print(f"\n✅  Solved in {turn + 1} guess(es)! Word was: {target}")
            return

    # ── Game over — didn't solve ──────────────────────────────────────────────
    if RICH:
        console.print()
        display_board_rich(guesses, feedbacks, target_hidden=False)
        console.print(
            f"\n[bold red]❌  Failed to solve in {MAX_GUESSES} guesses. "
            f"Word was: [underline]{target}[/][/]"
        )
    else:
        print(f"\n❌  Failed. Word was: {target}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_name: str,
    checkpoint: Optional[str],
    base_only:  bool,
) -> tuple:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if RICH:
        console.print(f"[dim]Loading base model: {model_name}[/]")
    else:
        print(f"Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = dtype,
        device_map  = "cuda",
        trust_remote_code = True,
    )

    if not base_only and checkpoint and os.path.isdir(checkpoint):
        try:
            from peft import PeftModel
            if RICH:
                console.print(f"[dim]Loading LoRA adapter: {checkpoint}[/]")
            else:
                print(f"Loading LoRA adapter: {checkpoint}")
            model = PeftModel.from_pretrained(model, checkpoint)
            model = model.merge_and_unload()
            if RICH:
                console.print("[green]✓ LoRA weights merged[/]")
        except Exception as e:
            if RICH:
                console.print(f"[yellow]⚠ Could not load LoRA ({e}) — using base model[/]")
            else:
                print(f"⚠ Could not load LoRA: {e}")
    elif not base_only and checkpoint:
        msg = f"Checkpoint not found at '{checkpoint}' — using base model (train first!)"
        if RICH:
            console.print(f"[yellow]⚠ {msg}[/]")
        else:
            print(f"⚠ {msg}")

    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Play one Wordle game with live candidate analysis"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace base model ID"
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to trained LoRA checkpoint directory"
    )
    parser.add_argument(
        "--base-only", action="store_true",
        help="Use the raw base model, skip LoRA loading"
    )
    parser.add_argument(
        "--target", default=None,
        help="Secret target word (default: random). Must be a 5-letter word."
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N,
        help=f"Number of top candidates to display per turn (default {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--reveal", action="store_true",
        help="Show the secret target word at the start"
    )
    parser.add_argument(
        "--temp", type=float, default=0.4,
        help="Generation temperature (default 0.4)"
    )
    args = parser.parse_args()

    # Validate / pick target
    if args.target:
        target = args.target.upper()
        if len(target) != 5 or not target.isalpha():
            sys.exit("❌  Target must be a 5-letter alphabetic word.")
    else:
        target = random.choice(WORDLE_WORDS)

    model, tokenizer = load_model(args.model, args.checkpoint, args.base_only)

    play_game(
        model     = model,
        tokenizer = tokenizer,
        target    = target,
        top_n     = args.top,
        reveal    = args.reveal,
        gen_temp  = args.temp,
    )


if __name__ == "__main__":
    main()
