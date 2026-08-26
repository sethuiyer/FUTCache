# Negative result: geometric MDL is not semantic safety

The existing labeled semantic-cache benchmark was evaluated in a strict
two-stage protocol:

1. Embed the 30 unlabeled query phrasings and select ε using only those vectors.
2. Reveal intent labels and score the frozen cache against the same frontier.

The existing grid and MDL codec were not changed after labels were revealed.
The MDL selector chose `ε=0.70` because its geometric description was shortest:

```text
7 representatives, 234,176 model bits,
84 assignment bits, 34,862 residual bits,
269,125 total bits
```

But the labeled semantic score at that radius was only 52.2% precision,
60.0% recall, F1 `0.558`, and 11 false-positive cross-intent merges. The
supervised F1 maximum on the same fixed frontier was ε=`0.55` (F1 `0.743`),
while the safest zero-false-positive point was ε=`0.45` (100% precision,
F1 `0.710`).

The result is not a codec failure. MDL answered the question it was given:

> At what ε is this embedding cloud cheapest to describe?

It did not answer the application-specific question:

> At what ε are two queries semantically interchangeable?

Therefore geometric compressibility and semantic substitutability are distinct
objectives. Plain geometric MDL is appropriate for geometric novelty,
compression, summarization, and sensor streams; it is not a semantic-safety
certificate for answer caching. The focused regression test
`geometric MDL is not semantic safety` preserves this boundary without
changing the codec.
