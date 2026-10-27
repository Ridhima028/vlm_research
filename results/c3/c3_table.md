# C3 Results: Qwen vs LLaVA

C3 evaluates prompt behaviour on dry tiles. For binary prompts, the observed
percentage of explicit Yes answers is a false-alarm proxy.

| model   | prompt_id      |   n |   yes_count |   no_count |   other_count |   yes_rate_pct |   no_rate_pct |   other_rate_pct |   mean_response_words |
|:--------|:---------------|----:|------------:|-----------:|--------------:|---------------:|--------------:|-----------------:|----------------------:|
| llava   | direct         | 150 |           3 |        147 |             0 |          2     |        98     |                0 |                  1    |
| qwen    | direct         | 150 |           5 |        145 |             0 |          3.333 |        96.667 |                0 |                  1    |
| llava   | evidence       | 150 |           0 |        150 |             0 |          0     |       100     |                0 |                  1    |
| qwen    | evidence       | 150 |          12 |        138 |             0 |          8     |        92     |                0 |                  1    |
| llava   | conservative   | 150 |           0 |        150 |             0 |          0     |       100     |                0 |                  1    |
| qwen    | conservative   | 150 |           0 |        150 |             0 |          0     |       100     |                0 |                  1    |
| llava   | leading_damage | 150 |           0 |          0 |           150 |          0     |         0     |              100 |                 49.01 |
| qwen    | leading_damage | 150 |           0 |          0 |           150 |          0     |         0     |              100 |                 56.53 |

## Interpretation and limitations
- `direct`, `evidence`, and `conservative`: Yes rate is calculated from explicit Yes answers.
- `leading_damage`: answers are free text; `other` means the response did not begin with Yes/No.
- The `leading_damage` Yes/No parser does not determine whether a response semantically claims flooding.
- Use a human-validated LLM judge for semantic classification before reporting its false-claim rate.
- Dry tiles alone cannot estimate recall on flood-positive images or overall model accuracy.
