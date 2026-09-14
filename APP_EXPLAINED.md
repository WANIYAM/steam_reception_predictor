# Explaining the Steam Reception Predictor — Your Presentation Script

This is written so you can either read it directly to an audience or use it as your own
speaking notes. Every technical term is defined the first time it shows up, in plain language,
right where you'll need it.

---

## Part 1: The 30-second pitch

> "We built a tool that looks at a game's basic details — price, genre, platform, who made it —
> and predicts two things: **will players like it**, and **how many people will end up owning
> it**. It learned these patterns by studying 26,564 real games that have already been released
> on Steam. There are two versions of the tool in one app: one for **game developers** planning
> a launch, and one for **players** looking for games worth their time."

---

## Part 2: Words you'll need (say these in plain language, audience won't need jargon)

Use this section as your own cheat-sheet — you don't need to read all of this aloud, just have
it ready in case someone asks "what does that mean?"

| Term | Plain-language definition |
|---|---|
| **Machine learning model** | A program that finds patterns in past examples and uses them to make guesses about new, similar situations. Think of it like a very experienced store clerk who's seen thousands of games launch and has developed a "gut feeling" for what tends to do well — except this gut feeling is built from actual data, not vibes. |
| **Training** | The process of showing the model thousands of past examples (in our case, real Steam games and how they actually did) so it can learn the patterns. This already happened — it's baked into the app, not something that happens live. |
| **Feature** | One piece of information the model is allowed to look at — e.g. "price," "is it an action game," "does it support Mac." We gave the model 55 features per game. |
| **Prediction** | The model's guess about something it wasn't told directly — e.g. "will this be well-received?" It's an educated guess based on patterns, not a certainty. |
| **Classification** | A prediction that's a category, like "Yes/No" or "Poor/Average/Great." |
| **Regression** | A prediction that's a number, like "73% of reviews will be positive" or "this will sell about 50,000 copies." |
| **Accuracy / ROC-AUC / R²** | Different ways of scoring "how good is the model, honestly?" You don't need the math — just know: **1.0 (or 100%) would mean perfect**, and **0.5 (or 50%) means no better than a coin flip**. Our numbers sit in a realistic middle ground (explained in Part 5) — good enough to be genuinely useful, not so high that we should be suspicious of it. |
| **Overfitting** | When a model basically memorizes the exact examples it was trained on instead of learning a general pattern — like a student who memorizes last year's exam answers instead of learning the subject, and then fails when the questions are worded differently. We specifically checked for and avoided this (see Part 5). |
| **Data leakage** | A sneaky bug where the model accidentally gets to see the answer while it's supposed to be guessing — like a student peeking at the answer key. We found and fixed one of these (explained in Part 5) — it's the kind of mistake that makes a model look great in testing but fail in the real world. |
| **CatBoost** | The specific brand of machine-learning technique we used to build these models. You can just call it "the model" or "the algorithm" — the audience doesn't need the brand name unless someone asks specifically. |

---

## Part 3: Walking through the app, screen by screen

### The sidebar (always visible on the left)

**What they see:** A menu to pick "Developer," "Player," or "About," plus a box showing the
model's accuracy numbers, plus a orange warning box.

**What to say:** *"On the left, you choose which version of the tool you want — are you a
developer planning a game, or a player looking for something to play? We also show our accuracy
numbers right up front — we didn't want to hide how good or limited this tool actually is."*

**The warning box** says the data only goes up to 2019. Explain: *"The games in our training
data stop in May 2019 — so this tool understands 'pre-2019 Steam' really well, but trends have
shifted since then. Think of it like asking someone who was really plugged into the gaming
industry up until 2019, then went silent — extremely knowledgeable, but not up to date on 2024's
biggest trends."*

---

### 🛠️ Developer Studio — Tab 1: "Reception Predictor"

**What they see:** A form — price, number of achievements, age rating, platforms, genres, key
features (like multiplayer or achievements), community tags, and then a section to pick a real
developer/publisher or leave it as "new/unknown." Press a button, get a prediction.

**What to say, step by step:**

1. *"You fill this out like you're describing a game you're thinking about making."*
2. *"Then — and this is the interesting part — you can optionally pick a **real, existing Steam
   studio** from a dropdown. If you pick a studio with a strong track record, watch what happens
   to the prediction versus picking 'new/unknown.'"*
   - **Why this matters (say this part slowly, it's the coolest part of the demo):** *"We found
     that who made the game is actually one of the single biggest predictors of success — even
     more than genre or price. A studio's past track record tells you a lot. So we built that
     into the tool directly: you can simulate 'what if a well-regarded studio made this exact
     game' versus 'what if a total unknown made it.' In our testing, that alone swung the
     prediction by over 30 percentage points on an otherwise identical game."*
3. *"After you hit predict, you get four things:"*
   - **A tier badge** (Poor / Average / Great) — the simple verdict.
   - **A percentage** — the predicted share of positive reviews.
   - **A probability** — how likely the model thinks it is to be "well-received" (its own
     70%-positive-reviews-or-better bar).
   - **An estimated owner count** — roughly how many people the model thinks will end up owning
     the game.
4. **Important honesty line to include:** *"None of these numbers are a promise. They're the
   model's best guess based on patterns in games that already exist. A brilliant, well-marketed
   game can beat the odds, and a mediocre one backed by a beloved studio can underperform them."*

---

### 🛠️ Developer Studio — Tab 2: "Price Sensitivity"

**What they see:** Two line charts — one showing predicted review score at every price point
from $0 to $60, one showing predicted number of owners at every price point.

**What to say:** *"This answers the question every developer actually agonizes over: what should
I charge? We hold everything else about your game fixed and just slide the price up and down, so
you can see the trade-off. Usually you'll notice these two charts don't peak at the same price —
cheaper (or free) games tend to reach more people, even when they don't score the highest reviews.
It's a classic reach-versus-reception trade-off, and now you can actually see the shape of it for
your specific game idea instead of guessing."*

---

### 🛠️ Developer Studio — Tab 3: "What Drives Success"

**What they see:** A bar chart of the top 15 things that matter most to the model, and a scatter
plot comparing genres by how well they typically do vs. how many people typically buy them.

**What to say about the bar chart:** *"This is the model showing its work — literally ranking
which pieces of information mattered most when it was learning. You'll notice developer/publisher
reputation and identity dominate the top of the list. After that, things like having cloud-save
support, controller support, and being single-player all show up as meaningful signals — which
lines up with intuition: well-supported, polished games tend to do better."*

**What to say about the scatter plot:** *"Each dot is a genre. Dots further right have historically
sold to more people; dots higher up have historically gotten better reviews. This is useful for
positioning — are you entering a genre that's crowded but well-loved, or one that's less common
but riskier?"*

---

### 🕹️ Player Corner — Tab 1: "Hidden Gems"

**What they see:** A scrollable list of real games with high review scores but low owner counts.

**What to say:** *"This flips the tool around for players. Instead of trusting Steam's own
popularity ranking — which rewards games that are already popular — we're specifically looking
for games that reviewers loved but that never got much traction. These are the under-the-radar
picks. You can filter by max price if you're looking for something cheap."*

---

### 🕹️ Player Corner — Tab 2: "Over/Underrated"

**What they see:** Two lists you can toggle between — games that did **better** than the model
expected, and games that did **worse** than the model expected.

**What to say:** *"For every game, our model made a prediction based on its price, genre, and
platform profile — basically 'here's what a typical game like this usually gets.' Then we compare
that prediction to what actually happened. When a game way outperforms what its profile would
predict, that's often a sign of something special — brilliant execution, incredible word of mouth,
a surprise hit. When a game way underperforms, that's often a sign something went wrong at
launch — bugs, controversy, broken promises. It's a genuinely different way to explore the
catalog than just sorting by star rating."*

*(Good moment for a concrete, relatable example if your audience knows gaming history — e.g. a
famous launch disaster would show up strongly in the "underperformed" list, because the game's
price/genre/platform profile alone would have predicted a normal reception, and it didn't get
one.)*

---

### 🕹️ Player Corner — Tab 3: "Browse & Compare"

**What they see:** Filters for genre, price range, and a minimum "model tier," producing a table
of matching games.

**What to say:** *"This is the straightforward, practical tool — filter down to what you're in
the mood for, and see the model's own quality tier alongside the real review data, so you can
sanity-check the model against reality yourself."*

---

## Part 4: If someone asks "how does it actually work" (a simple analogy)

> "Imagine you gave a very patient assistant a spreadsheet of 26,564 real Steam games — their
> price, genre, platform, achievements, who made them, and how they actually turned out. You
> asked that assistant to stare at that spreadsheet until they could predict, just from the first
> few columns, how the last column (the outcome) would turn out. That's what happened here,
> except the 'assistant' is a computer program that can hold far more patterns in its head than a
> person could, and can do it in a few minutes instead of a lifetime."

> "One important detail: it never got to see the actual answer while learning to predict a
> specific game — for every single game, we made sure it only ever learned from *other* games,
> never from that game's own outcome. Otherwise it would be like giving a student the answer key
> and being surprised when they ace the test — that would tell you nothing about whether they
> actually learned anything."

---

## Part 5: If someone pushes on accuracy — "is this actually good?"

Be upfront and confident about this — it's a strength, not a weakness, that these numbers are
honest.

- **Predicting whether players will like a game, using only its price/genre/platform, is
  genuinely hard.** Most of what makes a game good — how it feels to play, the writing, whether
  it's buggy, how it's marketed — isn't information we gave the model at all. So a model that
  was "99% accurate" on this exact task would be a red flag, not a triumph — it would almost
  certainly mean the model was cheating somehow (seeing information it shouldn't have).
- **Our numbers are realistic and were fixed once already.** Early versions of this project had
  a bug where a "developer reputation" feature accidentally used a game's own outcome to describe
  that same game — like grading yourself using your own answer key. We caught it, fixed it with
  a technique that only ever uses a studio's *other* games, and the accuracy numbers you see now
  are the honest, corrected ones.
- **We also specifically tried to improve accuracy and can show our work.** We tested several
  ideas — some worked (giving the model real studio names, not just an averaged score, was the
  single biggest improvement), some didn't (blending multiple model types together barely moved
  the needle, so we didn't add that complexity). That's a good story to tell: *"we didn't just
  build one thing and stop — we specifically tried to make it better and can show you what
  worked and what didn't."*

**One good soundbite if you need a single number to say out loud:**
*"For the simple yes/no version — will this be well-received — the model gets it right about
76-84% of the time on games it's never seen before, which is a solid, genuine signal, even
though it's obviously not a crystal ball."*

---

## Part 6: Likely audience questions and short answers

**"Can I trust the exact owner-count number?"**
No — treat it as a rough order of magnitude, not a precise forecast. Steam itself never
published exact owner counts for these games either (only ranges), so even the real data we
trained on is an estimate.

**"Does this work for games released this year?"**
Not reliably — the training data stops in 2019, so it doesn't know about anything that's changed
in the market since. It's a historical pattern-matcher, not a live trend tracker.

**"What would make it more accurate?"**
Feeding it richer information it doesn't currently have — the actual text of a game's store
description, or real community signals like wishlist counts and forum activity. Right now it
only sees structured facts like price and genre, not anything about the game's actual content or
quality of execution.

**"Is the model biased toward big studios?"**
It reflects what's true in the historical data — bigger, more established studios have, on
average, put out better-received games in this dataset. That's a real pattern, not something we
injected — but it's worth being upfront that it means the tool will naturally be more
optimistic about known studios and more neutral/cautious about newcomers, which mirrors a real
bias in how the games market itself works, not just in the model.
