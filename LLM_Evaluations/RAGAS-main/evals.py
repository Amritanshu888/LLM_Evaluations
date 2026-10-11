#!/usr/bin/env python
# coding: utf-8

# # Evaluation Suite
# ## How do we know our RAG system is actually working?
# 
# > **Goal:** Measure 6 failure modes using RAGAS on mock data.
# > Each experiment isolates **one specific thing** that can go wrong in a RAG pipeline.

# ## Why Evaluation Matters
# 
# A RAG pipeline has three places things can break:
# 
# | Layer | What can go wrong | Metric that catches it |
# |-------|-------------------|------------------------|
# | Retrieval | Fetches noisy / wrong chunks | Context Precision |
# | Retrieval | Misses key chunks | Context Recall |
# | Generation | Answer invents facts not in context | Faithfulness |
# | Generation | Answer is vague or off-topic | Answer Relevancy |
# | Generation | Answer has wrong facts | Answer Correctness |
# | Agent/Planner | Calls wrong tools | Tool Correctness |
# 
# Running all 6 gives you **full pipeline coverage** — retrieval, generation, and agent layers.

# ## Token Budget — Why We Use a Separate `JUDGE_GROQ` Key
# 
# RAGAS metrics work by asking an LLM judge to evaluate each sample.
# Each `evaluate()` call triggers multiple LLM requests:
# 
# | Metric | LLM calls / sample | ~Tokens / call | Calls for 3 samples |
# |--------|--------------------|----------------|---------------------|
# | Faithfulness | 2 | ~300 | 6 |
# | Answer Relevancy | 1 | ~250 | 3 |
# | Context Precision | 2 | ~350 | 6 |
# | Context Recall | 2 | ~350 | 6 |
# | Answer Correctness | 3 | ~300 | 9 |
# | **Total per experiment** | | | **~30 calls · ~9,000 tokens** |
# 
# **Groq free tier:** ~14,400 TPM for `llama-3.1-8b-instant`.
# We add **60-second cooldowns** between each `evaluate()` call to stay within limits.
# 
# > Add `JUDGE_GROQ=<second_key>` to `.env` so your main API key is never exhausted.
# > If `JUDGE_GROQ` is not set, the notebook falls back to `GROQ_API_KEY` automatically.

# In[7]:


import os
import asyncio
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI     # async client — required by ragas abatch_score



# In[8]:


# ── RAGAS ─────────────────────────────────────────────────────────────────────
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)


load_dotenv()
print("✅ All imports loaded")


# ENV SETUP

# In[10]:


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JUDGE_GROQ_API_KEY = os.getenv("JUDGE_GROQ", GROQ_API_KEY)


groq_client = AsyncOpenAI(
    api_key=JUDGE_GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

judge_llm = llm_factory("llama-3.1-8b-instant", provider="openai", client=groq_client)

# ── Local Embeddings (zero API cost) ──────────────────────────────────────────
ragas_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    use_api=False,
)

print(f"✅ Judge LLM  : {type(judge_llm).__name__}")
print(f"✅ Embeddings : {type(ragas_embeddings).__name__} (local — no API key needed)")
using = "JUDGE_GROQ" if os.getenv("JUDGE_GROQ") else "GROQ_API_KEY (fallback)"
print(f"✅ Using key  : {using}")


# HELPER FUNCTIONS

# In[11]:


async def async_cooldown(seconds=60):
    """Async cooldown — works inside Jupyter's running event loop."""
    print(f"\n⏳ Cooldown {seconds}s (Groq rate-limit buffer)...", end=" ")
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
        print(".", end="", flush=True)
    print("  ✅ Ready.\n")
    
    
# scores --  0 - 1

def show_scores(df, col, title):
    """Colour-coded score table for one metric column."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for i, row in df.iterrows():
        score = row.get(col, float("nan"))
        if score != score:
            bar = "⬜"
        elif score >= 0.75:
            bar = "🟢"
        elif score >= 0.5:
            bar = "🟡"
        else:
            bar = "🔴"
        q = str(row.get("user_input", f"Sample {i+1}"))[:55]
        print(f"  {bar}  {score:.2f}  |  {q}")
    avg = df[col].mean()
    label = "✅ Good" if avg >= 0.75 else ("⚠️  Fair" if avg >= 0.5 else "❌ Poor")
    print(f"{'─'*60}")
    print(f"  AVG: {avg:.2f}  {label}")
    print(f"{'='*60}\n")



# ## Metrics Reference — What Each Metric Tests
# 
# ### 1. All 6 Metrics at a Glance
# 
# | # | Metric | Library | What it measures | Score = 1.0 means |
# |---|--------|---------|-----------------|-------------------|
# | 1 | **Faithfulness** | RAGAS | Does the answer only use facts from the retrieved context? | Zero hallucination |
# | 2 | **Answer Relevancy** | RAGAS | Is the answer directly addressing the question asked? | Perfectly on-topic |
# | 3 | **Context Precision** | RAGAS | Are relevant chunks ranked *above* noisy chunks? | Best chunks ranked first |
# | 4 | **Context Recall** | RAGAS | Did retrieval fetch *all* facts the reference answer needs? | Nothing important missed |
# | 5 | **Answer Correctness** | RAGAS | Does the answer match the ground-truth reference? | Factually identical |
# | 6 | **Tool Correctness** | Pure Python (Jaccard) | Did the agent call the right tools (no more, no less)? | Exact tool match |
# 
# ---
# 
# ### 2. What Inputs Does Each Metric Need?
# 
# | Metric | `user_input` | `retrieved_contexts` | `response` | `reference` | LLM calls? | Embeddings? |
# |--------|:---:|:---:|:---:|:---:|:---:|:---:|
# | Faithfulness | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
# | Answer Relevancy | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
# | Context Precision | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
# | Context Recall | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
# | Answer Correctness | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
# | Tool Correctness | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
# 
# > ⚡ **Tool Correctness** is fully deterministic — zero LLM calls, zero cost, no API key needed.
# 
# 
# ---
# 
# ### 3. Which Pipeline Layer?
# 
# ```
#   ┌─────────────────┬─────────────────┬──────────────────┐
#   │   RETRIEVAL     │   GENERATION    │  AGENT/PLANNER   │
#   ├─────────────────┼─────────────────┼──────────────────┤
#   │ Context         │ Faithfulness    │ Tool             │
#   │ Precision       │                 │ Correctness      │
#   │                 │ Answer          │                  │
#   │ Context         │ Relevancy       │                  │
#   │ Recall          │                 │                  │
#   │                 │ Answer          │                  │
#   │                 │ Correctness     │                  │
#   └─────────────────┴─────────────────┴──────────────────┘
# ```
# 
# ---
# 
# ### 4. Score Thresholds
# 
# | Score | Verdict | What to do |
# |-------|---------|------------|
# | ≥ 0.75 | 🟢 Good | Ship it |
# | 0.50 – 0.75 | 🟡 Fair | Investigate |
# | < 0.50 | 🔴 Poor | Fix before shipping |
# 
# ---
# 
# ### 5. Metric → Code (exact constructors used in this notebook)
# 
# ```python
# Faithfulness(llm=judge_llm)                                    # no embeddings
# AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)    # needs embeddings
# ContextPrecision(llm=judge_llm)                                # no embeddings
# ContextRecall(llm=judge_llm)                                   # no embeddings
# AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)  # needs embeddings
# 
# tool_correctness_score(case)                                   # no LLM at all
# ```

# ---
# ## Experiment 1 — Faithfulness (Hallucination Detection)
# **Domain:** 🛒 E-commerce Customer Support
# 
# ### What are we testing?
# A customer asks about the return/refund policy. The RAG system retrieves policy documents.
# We check: *does the answer only use facts from those documents, or does it add things never mentioned?*
# 
# ### The 3 Scenarios
# 
# | Sample | What happens | Expected score |
# |--------|-------------|----------------|
# | **Good** | Answer stays within the retrieved context | 🟢 High |
# | **Medium** | Answer adds a made-up 10% coupon not in any chunk | 🟡 Lower |
# | **Bad** | Answer directly contradicts the context (says free returns when context says paid) | 🔴 Low |
# 
# > **Faithfulness** = fraction of answer sentences that can be verified against the retrieved context.

# In[12]:


# ── Scenario: E-commerce Returns ─────────────────────────────────────────────
hall_samples = [
    # Good: answer is grounded in context only
    SingleTurnSample(
        user_input="Can I return a used laptop?",
        retrieved_contexts=[
            "Our return policy allows returns within 30 days of purchase for unused, unopened electronics.",
            "Items must be in original packaging with all accessories included.",
        ],
        response="According to the return policy, laptops must be unused and in original packaging to be eligible for a return within 30 days.",
        reference="No. Only unused, unopened electronics in original packaging qualify for returns within 30 days.",
    ),
    # Medium: response adds a fabricated 10% coupon not in any context chunk
    SingleTurnSample(
        user_input="What is the refund timeline?",
        retrieved_contexts=[
            "Approved refunds are processed within 5 to 7 business days.",
        ],
        response="Refunds are processed in 5 to 7 business days and you will also receive a 10%\ discount coupon on your next order.",
        reference="Refunds are processed within 5 to 7 business days.",
    ),
    # Bad: complete contradiction — context says customer pays, answer says free
    SingleTurnSample(
        user_input="Do you offer free return shipping?",
        retrieved_contexts=[
            "Customers are responsible for return shipping costs unless the item is defective.",
        ],
        response="Yes, we provide free prepaid return shipping labels for all returns including buyer remorse returns.",
        reference="No. Customers pay return shipping unless the item is defective.",
    ),
]

print("Running Exp 1 — Faithfulness (E-commerce Returns)...")
_m = Faithfulness(llm=judge_llm)
_inputs = [
    {"user_input": s.user_input, "response": s.response, "retrieved_contexts": s.retrieved_contexts}
    for s in hall_samples
]
_res = await _m.abatch_score(_inputs)
df_hall = pd.DataFrame([
    {"user_input": s.user_input, "faithfulness": float(r.value)}
    for s, r in zip(hall_samples, _res)
])
show_scores(df_hall, "faithfulness", "Exp 1 — Faithfulness (E-commerce Returns)")
await async_cooldown(60)


# ---
# ## Experiment 2 — Answer Relevancy (Vague / Off-Topic Answers)
# **Domain:** ✈️ Travel Planning Assistant
# 
# ### What are we testing?
# A traveller asks specific questions about Japan. The context has the right information.
# We check: *does the answer address what was asked, or is it vague and generic?*
# 
# ### The 3 Scenarios
# 
# | Sample | What happens | Expected score |
# |--------|-------------|----------------|
# | **Good** | Direct, specific answer that addresses the exact question | 🟢 High |
# | **Medium** | Vague answer that talks around the topic but never answers it | 🟡 Lower |
# | **Bad** | Completely off-topic — asked about rail passes, answered about food | 🔴 Low |
# 
# > **Answer Relevancy** scores how well the response answers the *specific question asked* — not just quality of writing.
# > Uses embeddings to compare question vs answer semantically.

# In[13]:


rel_samples = [
    # Good: directly and specifically answers the visa question
    SingleTurnSample(
        user_input="What are the visa requirements for visiting Japan from India?",
        retrieved_contexts=[
            "Indian passport holders require a tourist visa to enter Japan.",
            "Visa applications must be submitted at the Japanese embassy with a valid passport, photos, and a travel itinerary.",
        ],
        response="Indian citizens need a tourist visa to enter Japan. You apply at the Japanese embassy with your passport, recent photos, and a travel itinerary.",
        reference="Indian passport holders need a tourist visa for Japan. Apply at the Japanese embassy with passport, photos, and itinerary.",
    ),
    # Medium: vague — fails to mention any specific places
    SingleTurnSample(
        user_input="What are the best places to visit in Japan in April?",
        retrieved_contexts=[
            "April is cherry blossom season in Japan. Popular spots include Ueno Park in Tokyo, Maruyama Park in Kyoto, and Hirosaki Castle in Aomori.",
        ],
        response="Japan is a wonderful country with many beautiful places to visit. There are so many things to see and do throughout the year.",
        reference="In April, top cherry blossom spots are Ueno Park (Tokyo), Maruyama Park (Kyoto), and Hirosaki Castle (Aomori).",
    ),
    # Bad: off-topic — asked about buying a rail pass, answered about food
    SingleTurnSample(
        user_input="How do I buy a Japan Rail Pass?",
        retrieved_contexts=[
            "The Japan Rail Pass can be purchased online before traveling or at major JR stations on arrival.",
            "It covers unlimited travel on most JR trains including the Shinkansen bullet train.",
        ],
        response="Japanese cuisine is incredible. Must-try dishes include sushi, ramen, and tempura during your visit.",
        reference="You can buy a Japan Rail Pass online before your trip or at major JR stations in Japan.",
    ),
]


print("Running Exp 2 — Answer Relevancy (Travel Planning)...")
_m = AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings)
_inputs = [
    {"user_input": s.user_input, "response": s.response}
    for s in rel_samples
]
_res = await _m.abatch_score(_inputs)
df_rel = pd.DataFrame([
    {"user_input": s.user_input, "answer_relevancy": float(r.value)}
    for s, r in zip(rel_samples, _res)
])
show_scores(df_rel, "answer_relevancy", "Exp 2 — Answer Relevancy (Travel Planning)")
await async_cooldown(60)


# ---
# ## Experiment 3 — Context Precision (Noisy Retrieval)
# **Domain:** 💻 Python Coding Tutor
# 
# ### What are we testing?
# A student asks a basic Python question. The retriever fetches multiple chunks — some useful, some random.
# We check: *are the useful chunks ranked at the **top**, or are they buried under irrelevant noise?*
# 
# ### The 3 Scenarios
# 
# | Sample | What happens | Expected score |
# |--------|-------------|----------------|
# | **Good** | Relevant chunks appear first, noise is at the bottom | 🟢 High |
# | **Medium** | One irrelevant chunk appears before the useful one | 🟡 Lower |
# | **Bad** | Two irrelevant chunks appear before the single useful chunk | 🔴 Low |
# 
# > **Context Precision** penalises retrieval systems that return the right document in the wrong position.
# > Fix: adjust similarity threshold, add a re-ranker, or reduce top-k.

# In[14]:


# ── Scenario: Python Coding Help ─────────────────────────────────────────────
prec_samples = [
    # Good: relevant chunks first, noise last
    SingleTurnSample(
        user_input="How do I read a CSV file in Python?",
        retrieved_contexts=[
            "Use pandas: import pandas as pd; df = pd.read_csv('file.csv') to load a CSV into a DataFrame.",
            "pd.read_csv() accepts parameters like sep, header, encoding, and index_col for customisation.",
            "Python lists are ordered, mutable sequences defined with square brackets.",
        ],
        response="To read a CSV in Python, use pandas: import pandas as pd; df = pd.read_csv('file.csv'). You can also set sep, header, and encoding.",
        reference="Use pandas pd.read_csv('file.csv') to read a CSV file into a DataFrame.",
    ),
    # Medium: one irrelevant chunk before the useful one
    SingleTurnSample(
        user_input="How do I reverse a string in Python?",
        retrieved_contexts=[
            "Python supports various data types: int, float, string, list, and dict.",
            "To reverse a string, use slicing: s[::-1] or use ''.join(reversed(s)).",
        ],
        response="You can reverse a string using slicing: s[::-1].",
        reference="Reverse a string with s[::-1] or ''.join(reversed(s)).",
    ),
    # Bad: two irrelevant chunks before the single useful chunk
    SingleTurnSample(
        user_input="How do I sort a list in Python?",
        retrieved_contexts=[
            "Python decorators wrap functions to add behaviour using the @ symbol.",
            "The Global Interpreter Lock (GIL) affects multithreaded Python performance.",
            "Use list.sort() for in-place sorting or sorted(list) for a new sorted list.",
        ],
        response="Use list.sort() to sort in-place or sorted(list) to get a new sorted list.",
        reference="Sort a list with list.sort() (in-place) or sorted(list) (returns new list).",
    ),
]

print("Running Exp 3 — Context Precision (Python Coding)...")
_m = ContextPrecision(llm=judge_llm)
_inputs = [
    {"user_input": s.user_input, "reference": s.reference, "retrieved_contexts": s.retrieved_contexts}
    for s in prec_samples
]
_res = await _m.abatch_score(_inputs)
df_prec = pd.DataFrame([
    {"user_input": s.user_input, "context_precision": float(r.value)}
    for s, r in zip(prec_samples, _res)
])
show_scores(df_prec, "context_precision", "Exp 3 — Context Precision (Python Coding)")
await async_cooldown(60)


# ---
# ## Experiment 4 — Context Recall (Incomplete Retrieval)
# **Domain:** 🍳 Recipe / Cooking Assistant
# 
# ### What are we testing?
# A user asks for a recipe. The retriever should fetch ALL the steps needed to make the dish.
# We check: *did the retriever miss any key information that the reference answer requires?*
# 
# ### The 3 Scenarios
# 
# | Sample | What happens | Expected score |
# |--------|-------------|----------------|
# | **Good** | Context contains all 4 key steps of the recipe | 🟢 High |
# | **Medium** | Context is missing the oven temperature and bake time | 🟡 Lower |
# | **Bad** | Context contains only pasta trivia — no cooking instructions at all | 🔴 Low |
# 
# > **Context Recall** compares what is in the context against what the *reference answer* requires.
# > Low recall = your retriever needs better chunking, more sources, or a larger top-k.

# In[15]:


# ── Scenario: Cooking / Recipes ──────────────────────────────────────────────
recall_samples = [
    # Good: all 4 steps present in context
    SingleTurnSample(
        user_input="How do I make a classic French omelette?",
        retrieved_contexts=[
            "Crack 3 eggs into a bowl and whisk until smooth. Season with salt and pepper.",
            "Heat butter in a non-stick pan over medium heat until foamy but not brown.",
            "Pour the eggs in and stir with a spatula while shaking the pan for 30 seconds.",
            "Fold the omelette into thirds and slide onto a plate — it should be pale yellow, not browned.",
        ],
        response="Whisk 3 eggs with salt and pepper. Melt butter until foamy, pour eggs in, stir and shake for 30 seconds, fold into thirds. Serve pale yellow.",
        reference="Whisk 3 eggs, season, cook in foamy butter over medium heat stirring constantly, fold into thirds. The omelette should stay pale, not browned.",
    ),
    # Medium: missing oven temp and bake time for banana bread
    SingleTurnSample(
        user_input="What are the steps to make banana bread?",
        retrieved_contexts=[
            "Mash 3 ripe bananas in a mixing bowl.",
            "Mix in 75g melted butter, 150g sugar, 1 egg, and 1 tsp vanilla extract.",
        ],
        response="Mash the bananas, mix with butter, sugar, egg, and vanilla, pour into a loaf tin and bake until golden.",
        reference="Mash 3 bananas, mix with butter, sugar, egg, vanilla, fold in 190g flour with baking soda and salt. Bake at 175 C for 60 minutes.",
    ),
    # Bad: context has nothing relevant to cooking pasta
    SingleTurnSample(
        user_input="How long should I cook spaghetti?",
        retrieved_contexts=[
            "Pasta was introduced to Italy from Arab traders in the 12th century.",
            "There are over 350 recognised pasta shapes in Italian culinary tradition.",
        ],
        response="Cook pasta according to the package instructions until al dente.",
        reference="Boil spaghetti in salted water for 8 to 10 minutes. Taste one strand 1 to 2 minutes before the package time for al dente texture.",
    ),
]


print("Running Exp 4 — Context Recall (Cooking / Recipes)...")
_m = ContextRecall(llm=judge_llm)
_inputs = [
    {"user_input": s.user_input, "retrieved_contexts": s.retrieved_contexts, "reference": s.reference}
    for s in recall_samples
]
_res = await _m.abatch_score(_inputs)
df_recall = pd.DataFrame([
    {"user_input": s.user_input, "context_recall": float(r.value)}
    for s, r in zip(recall_samples, _res)
])
show_scores(df_recall, "context_recall", "Exp 4 — Context Recall (Cooking / Recipes)")
await async_cooldown(60)


# ---
# ## Experiment 5 — Answer Correctness (Wrong Facts)
# **Domain:** 🌍 General Knowledge Q&A
# 
# ### What are we testing?
# A user asks factual questions everyone knows — science, history, geography.
# We check: *does the RAG answer match the ground-truth reference, or does it contain wrong information?*
# 
# ### The 3 Scenarios
# 
# | Sample | What happens | Expected score |
# |--------|-------------|----------------|
# | **Good** | Answer is factually correct and matches the reference | 🟢 High |
# | **Medium** | Answer is on the right topic but gives the wrong number (90 C instead of 100 C) | 🟡 Lower |
# | **Bad** | Answer names the completely wrong thing (Venus instead of Mercury) | 🔴 Low |
# 
# > **Answer Correctness** combines factual overlap AND semantic similarity against the reference.
# > Use this whenever you have ground-truth answers and want to catch factual errors.

# In[16]:


# ── Scenario: General Knowledge ──────────────────────────────────────────────
correct_samples = [
    # Good: fully correct, matches reference
    SingleTurnSample(
        user_input="Who invented the telephone?",
        retrieved_contexts=[
            "Alexander Graham Bell is credited with patenting the first practical telephone in 1876.",
            "Bell demonstrated it by speaking to his assistant Thomas Watson in the next room.",
        ],
        response="Alexander Graham Bell invented the telephone and received the patent in 1876.",
        reference="Alexander Graham Bell invented the telephone, patenting it in 1876.",
    ),
    # Medium: right topic, wrong number
    SingleTurnSample(
        user_input="What is the boiling point of water at sea level?",
        retrieved_contexts=[
            "Water boils at 100 degrees Celsius (212 F) at sea level under standard atmospheric pressure.",
        ],
        response="Water boils at 90 degrees Celsius at sea level.",
        reference="Water boils at 100 degrees Celsius (212 F) at sea level.",
    ),
    # Bad: completely wrong answer
    SingleTurnSample(
        user_input="Which planet is closest to the Sun?",
        retrieved_contexts=[
            "Mercury is the closest planet to the Sun, orbiting at an average distance of 57.9 million km.",
        ],
        response="Venus is the closest planet to the Sun.",
        reference="Mercury is the closest planet to the Sun.",
    ),
]

print("Running Exp 5 — Answer Correctness (General Knowledge)...")
_m = AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings)
_inputs = [
    {"user_input": s.user_input, "response": s.response, "reference": s.reference}
    for s in correct_samples
]
_res = await _m.abatch_score(_inputs)
df_correct = pd.DataFrame([
    {"user_input": s.user_input, "answer_correctness": float(r.value)}
    for s, r in zip(correct_samples, _res)
])
show_scores(df_correct, "answer_correctness", "Exp 5 — Answer Correctness (General Knowledge)")


# In[ ]:





# ---
# ## Experiment 6 — Tool Correctness (Wrong Tool Calls)
# **Domain:** 🏠 Smart Home Voice Assistant (Agent)
# 
# ### What are we testing?
# A user gives a voice command to a smart home assistant agent. The agent must call the right tool.
# We check: *did the agent call the correct tool with the correct parameters?*
# 
# ### The 3 Scenarios
# 
# | Sample | Command | Agent did | Expected |
# |--------|---------|-----------|----------|
# | **Good** | Turn on living room lights | Called `control_smart_light` ✅ | `control_smart_light` |
# | **Medium** | Play jazz in the kitchen | Called `control_smart_light` ❌ | `play_music` |
# | **Bad** | Set thermostat to 22 degrees | Called `set_thermostat` + extra `send_notification` ❌ | Just `set_thermostat` |
# 
# ### How the score works (Jaccard overlap — no LLM needed)
# 
# ```
# score = |called ∩ expected| / |called ∪ expected|
# ```
# 
# | Scenario | called | expected | score |
# |----------|--------|----------|-------|
# | Good | {control_smart_light} | {control_smart_light} | 1/1 = **1.0** 🟢 |
# | Medium | {control_smart_light} | {play_music} | 0/2 = **0.0** 🔴 |
# | Bad | {set_thermostat, send_notification} | {set_thermostat} | 1/2 = **0.5** 🟡 |
# 
# > ⚡ **Zero LLM calls, zero cost, no API key needed.**
# > Tool correctness is structural — we compare tool names, not semantics.

# In[17]:


# ── Simple dataclasses (no DeepEval dependency) ────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import List

@dataclass
class ToolCall:
    name: str
    input_parameters: dict = field(default_factory=dict)

@dataclass
class AgentTestCase:
    input: str
    actual_output: str
    tools_called: List[ToolCall] = field(default_factory=list)
    expected_tools: List[ToolCall] = field(default_factory=list)


# ── Scenario: Smart Home Agent ─────────────────────────────────────────────────────
tool_cases = [
    # Good: correct tool called
    AgentTestCase(
        input="Turn on the living room lights",
        actual_output="I have turned on the living room lights for you.",
        tools_called=[
            ToolCall(name="control_smart_light", input_parameters={"room": "living room", "action": "on"}),
        ],
        expected_tools=[ToolCall(name="control_smart_light")],
    ),
    # Medium: wrong tool — user asked for music, agent triggered lights
    AgentTestCase(
        input="Play jazz music in the kitchen",
        actual_output="Playing jazz in the kitchen.",
        tools_called=[
            ToolCall(name="control_smart_light", input_parameters={"room": "kitchen", "action": "on"}),
        ],
        expected_tools=[ToolCall(name="play_music")],
    ),
    # Bad: right primary tool but also called an extra unnecessary tool
    AgentTestCase(
        input="Set the thermostat to 22 degrees",
        actual_output="Setting thermostat to 22 degrees Celsius.",
        tools_called=[
            ToolCall(name="set_thermostat", input_parameters={"temperature": 22}),
            ToolCall(name="send_notification", input_parameters={"message": "Thermostat set to 22 C"}),
        ],
        expected_tools=[ToolCall(name="set_thermostat")],
    ),
]

def tool_correctness_score(case):
    """Jaccard: |called ∩ expected| / |called ∪ expected|.
    Penalises both missing tools (recall) and extra tools (precision).
    Zero LLM calls — purely structural comparison of tool names."""
    called   = {t.name for t in (case.tools_called  or [])}
    expected = {t.name for t in (case.expected_tools or [])}
    if not expected:
        return 1.0
    union = len(called | expected)
    return len(called & expected) / union if union > 0 else 0.0


print("Running Exp 6 — Tool Correctness (Smart Home Agent)...")
print("Note: Jaccard score — zero LLM calls, no API key needed.\n")

results_tool = []
for case in tool_cases:
    score = tool_correctness_score(case)
    results_tool.append({"user_input": case.input, "tool_correctness": score})

df_tool = pd.DataFrame(results_tool)
show_scores(df_tool, "tool_correctness", "Exp 6 — Tool Correctness (Smart Home Agent)")

