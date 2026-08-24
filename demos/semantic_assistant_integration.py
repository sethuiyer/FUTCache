#!/usr/bin/env python3
"""demos/semantic_assistant_integration.py — wire PackCache into a real app.

A realistic end-to-end integration: a support assistant that answers user
questions via a (mock) LLM over a 45-sentence knowledge base, with a
`PackCache` semantic cache in front of it. A repeat or rephrased question is
served from cache (bypassing the LLM) via `get_or_compute`, with TTL and an
LRU cap.

Measures (honest, on real embeddings):
  * cache hit rate and LLM calls avoided
  * cost saved (input+output tokens at a stated price)
  * latency: cached vs cold
  * precision: P(a served answer belongs to the SAME intent) -- a cross-intent
    merge (epsilon too large) serves the WRONG intent's answer

Run:  python demos/semantic_assistant_integration.py [--epsilon 0.45]
                                                       [--repeats 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from futcache import PackCache

MODEL = "hotchpotch/bekko-embedding-v1-a8m"
COST_PER_1M_OUTPUT = 15.00
COST_PER_1M_INPUT = 3.00
AVG_IN = 200
AVG_OUT = 120
LLM_LATENCY_S = 0.20

CORPUS = [
    "You can reset your password from the account settings page.",
    "Password resets require a confirmation link sent to your email.",
    "Choose a password with at least eight characters and a number.",
    "A password reset link expires after sixty minutes.",
    "To sign out, open the user menu and select logout.",
    "Logging out clears your local session on the device.",
    "Subscriptions are managed in the billing section.",
    "You can cancel a subscription at the end of the billing cycle.",
    "Cancelling stops renewals but keeps access until the period ends.",
    "Refunds are issued within five to ten business days.",
    "Eligible refunds are processed back to your original payment method.",
    "API keys are generated under the developer settings tab.",
    "Keep your API key secret; do not commit it to a repository.",
    "Change your email address from the profile settings.",
    "A verification email is sent to the new address.",
    "Invoices are available for download in the billing history.",
    "Each invoice includes a line-by-line summary of charges.",
    "A mobile app is available for iOS and Android.",
    "The mobile app supports the same features as the web version.",
    "There is a free tier with a monthly usage limit.",
    "Paid plans unlock higher usage caps and priority support.",
    "Two-factor authentication adds a second step at sign in.",
    "Enable 2FA from the security settings on your account.",
    "Authenticator apps generate the verification code.",
    "You can invite teammates from the organization settings.",
    "Audit logs record changes made to your workspace.",
    "Data is encrypted at rest and in transit.",
    "You can export your data in a CSV file.",
    "The help center has step-by-step guides for common tasks.",
]

# intent -> (canonical question, list of rephrasings)
QUESTIONS = {
    "password_reset": ["How do I reset my password?",
                       "What is the procedure to change my password?",
                       "I forgot my password, how can I get a new one?"],
    "sign_out": ["How do I log out of my account?",
                 "What is the way to sign out?",
                 "How can I end my session?"],
    "cancel_subscription": ["How do I cancel my subscription?",
                            "What is the process to cancel my plan?",
                            "I want to stop my subscription, how?"],
    "refund": ["How do I get a refund?",
               "What is the refund process?",
               "Can I get my money back, and how?"],
    "api_key": ["Where do I find my API key?",
                "How do I generate an API key?",
                "How do I get my API credentials?"],
    "change_email": ["How do I change my email address?",
                     "What is the way to update my email?",
                     "How can I modify the email on my account?"],
    "invoice": ["Where can I download my invoice?",
                "How do I get a copy of my bill?",
                "How do I access my billing statement?"],
    "mobile_app": ["Is there a mobile app?",
                   "Do you have an iOS or Android app?",
                   "Can I use this from my phone?"],
    "pricing": ["How much does it cost?",
                "What are the pricing tiers?",
                "What is the monthly price?"],
    "two_factor": ["How do I enable 2FA?",
                   "How do I start two-factor authentication?",
                   "How do I set up multifactor authentication?"],
}


class SupportAssistant:
    """A PackCache-backed semantic answer cache in front of an LLM."""

    def __init__(self, model, dim=384, eps=0.45, ttl=0.0, max_entries=0):
        self.model = model
        self.eps = eps
        self.corpus_vecs = model.encode(CORPUS, normalize_embeddings=True)
        self.cache = PackCache(dim, eps, distance="cosine",
                               domain_min=-1.0, domain_max=1.0,
                               backend="vptree", ttl=ttl,
                               max_entries=max_entries)

    def _answer(self, question, intent):
        """The 'LLM': retrieve the best knowledge sentence and answer."""
        time.sleep(LLM_LATENCY_S)  # simulated network + generation latency
        qv = self.model.encode([question], normalize_embeddings=True)[0]
        sim = self.corpus_vecs @ qv
        best = int(np.argmax(sim))
        # embed the intent so a served answer can be attributed to its intent
        return f"<intent:{intent}> {CORPUS[best]}"

    def ask(self, question, intent):
        """Return (answer, was_cached). Uses get_or_compute so a repeat or
        rephrase is served from cache and skips the LLM."""
        qv = self.model.encode([question], normalize_embeddings=True)[0]

        def compute(_p, intent=intent):
            return self._answer(question, intent).encode()

        answer, res = self.cache.get_or_compute(qv, compute, radius=self.eps)
        return answer.decode(), (not res.is_novel), res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epsilon", type=float, default=0.45)
    ap.add_argument("--ttl", type=float, default=0.0)
    ap.add_argument("--max-entries", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=4,
                    help="times to repeat each question stream")
    args = ap.parse_args()

    print("=" * 72)
    print("PackCache integration: semantic support assistant")
    print("=" * 72)
    print(f"  model: {MODEL} (384-d, L2-normalised)")
    print(f"  knowledge base: {len(CORPUS)} sentences; "
          f"{len(QUESTIONS)} intents x 3 phrasings")
    print(f"  epsilon={args.epsilon}  ttl={args.ttl}s  "
          f"max_entries={args.max_entries}  repeats={args.repeats}")
    print("  loading model ...")
    model = SentenceTransformer(MODEL)
    assistant = SupportAssistant(model, eps=args.epsilon, ttl=args.ttl,
                                 max_entries=args.max_entries)

    cost_per_call = (AVG_IN * COST_PER_1M_INPUT +
                     AVG_OUT * COST_PER_1M_OUTPUT) / 1e6

    # Build the stream: shuffle phrasings, repeat the whole thing.
    phrasings = [(intent, q) for intent, qs in QUESTIONS.items()
                 for q in qs]
    rng = np.random.default_rng(11)
    order = rng.permutation(len(phrasings))
    stream = [phrasings[i] for i in order]
    stream = stream * args.repeats

    hits = misses = correct = served = 0
    cold_s = 0.0
    cached_s = 0.0
    wrong = []
    for intent, question in stream:
        t0 = time.perf_counter()
        answer, was_cached, res = assistant.ask(question, intent)
        dt = time.perf_counter() - t0
        if was_cached:
            hits += 1
            cached_s += dt
        else:
            misses += 1
            cold_s += dt
        # attribute the served answer to an intent (embedded in the answer)
        served_intent = answer.split(">")[0].replace("<intent:", "")
        if served_intent == intent:
            correct += 1
        else:
            wrong.append((intent, served_intent))
        served += 1

    n = len(stream)
    naive_cost = n * cost_per_call
    cached_cost = misses * cost_per_call
    saved = naive_cost - cached_cost
    precision = correct / served

    print("\n[ Integration results ]")
    print(f"  queries: {n:,}   cold LLM calls: {misses:,} ({misses/n:.1%})"
          f"   cache hits: {hits:,} ({hits/n:.1%})")
    print(f"  avg latency: cold={cold_s/max(misses,1)*1e3:.1f} ms  "
          f"cached={cached_s/max(hits,1)*1e6:.1f} us")
    print(f"  reuse correctness: {correct}/{served} = {precision:.1%}")
    print(f"  cost: naive=${naive_cost:.2f}  with cache=${cached_cost:.2f}  "
          f"saved=${saved:.2f} ({saved/naive_cost:.1%})")
    if wrong:
        print(f"  cross-intent serves: {[(a, b) for a, b in wrong[:4]]}")
    print("\n[ Example transcript ]")
    for intent, question in stream[:6]:
        answer, was_cached, _ = assistant.ask(question, intent)
        tag = "CACHE" if was_cached else "LLM  "
        print(f"  [{tag}] {question[:52]:54s} -> {answer.split('> ')[-1][:46]}")
    print("  (epsilon too large would merge intents and serve the wrong answer;")
    print("   reuse correctness is the honest precision measure.)")
    print("  (cached latency includes the per-query embedding, which is unavoidable;")
    print("   the cache lookup itself is microseconds -- the LLM call is what is skipped.)")


if __name__ == "__main__":
    main()
