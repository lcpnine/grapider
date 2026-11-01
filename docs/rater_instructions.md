# Rater instructions

You are helping check whether one concept is a prerequisite of another. Your answers are used to measure how accurate a generated prerequisite graph is. There are no trick questions, and your own judgment is what we need.

## The question

Each row names two concepts, A and B. For each row, ask:

> **Would a course on B have to teach or assume A?**

| Answer | Use it when |
|---|---|
| `yes` | A course on B would have to teach A or assume students know it. |
| `no` | A course on B could be taught properly without A. |
| `unsure` | You can't decide, or you don't know one of the concepts well enough. |

Three rules:

- **Indirect prerequisites count.** If A is needed for something that B needs, answer `yes`, even if A is several steps back.
- **A broader topic counts only if a learner must understand it first.** "Machine learning" is not automatically a prerequisite of a specific method just because the method belongs to that field.
- **Being related is not enough.** Answer `no` if A and B are only often taught together, or if A is merely useful background. Also answer `no` if the direction is backwards, meaning A builds on B.

Examples from outside this topic area:

| A | B | Answer | Why |
|---|---|---|---|
| fractions | adding unlike fractions | `yes` | B can't be taught without A. |
| adding unlike fractions | fractions | `no` | The direction is backwards. |
| photosynthesis | cellular respiration | `no` | Related, but neither requires the other. |

## How to work

- Work alone. Don't discuss items with the other rater until both of you have finished.
- **Don't use ChatGPT, Claude or any other AI tool**, for answers or for explanations. These ratings are meant to be an independent human check on an AI system. A quick look at a textbook or Wikipedia to recall what a term means is fine.
- Answer every row if you can. Use `unsure` rather than leaving a row blank.
- Go with your first considered judgment, and don't try to guess how many answers should be `yes`.
- Some concept names are odd or vague, such as "others" or "course introduction". Answer as best you can, or use `unsure`.

## What to fill in

Use a separate blank copy of each sheet, and fill in only the column assigned to you. The other rater's column stays empty.

| Sheet | Rows | Purpose |
|---|---|---|
| `review_packet.csv` | 119 | Estimates how accurate the generated edges are. |
| `kappa_review.csv` | 50 | Checks pairs where existing labels are in question. |

Allow about 30–45 minutes in total. Type only `yes`, `no` or `unsure` in each cell.

## For the coordinator

1. Find two raters with a computer science or machine learning background, who haven't worked on this project and haven't seen its results.
2. Give each rater these instructions and blank copies of both sheets. Rater 1 fills the `rater_1_A_needed_for_B` columns, and rater 2 fills the `rater_2_A_needed_for_B` columns.
3. Don't share the labels, the scores, `runs/review_ratings.json` or anything in `runs/private/`. All of these would break the blind review.
4. Paste each rater's filled column into the matching column of `runs/review_packet.csv` and `runs/kappa_review.csv`, keeping the rows in the same order.
5. Run `.venv/bin/python -m prereq.experiment score_reviews`. It reports each rater's scores and their agreement under `human_review`, kept apart from the AI review under `ai_review`.

The packet is a uniform sample of what the generator emitted, so it can estimate precision. The second sheet was selected to include label disputes, so it only shows who is right on those disputes and isn't a benchmark.
