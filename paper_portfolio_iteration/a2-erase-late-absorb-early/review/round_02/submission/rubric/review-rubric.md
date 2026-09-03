# ICLR 2026 Review Rubric

Use this rubric for internal pre-submission review of an ICLR paper. It mirrors the public ICLR 2026 review form and reviewer guidance. It is not an acceptance predictor.

## Primary Sources

- ICLR 2026 Reviewer Guide: https://iclr.cc/Conferences/2026/ReviewerGuide
- ICLR 2026 Author Guide: https://iclr.cc/Conferences/2026/AuthorGuide
- ICLR 2026 Code of Ethics: https://iclr.cc/public/CodeOfEthics
- Public ICLR 2026 official-review example exposing Soundness, Presentation, Contribution, Rating, and Confidence: https://openreview.net/forum?id=1JZuEDq62N
- ICLR 2026 process retrospective: https://blog.iclr.cc/2026/03/31/a-retrospective-on-the-iclr-2026-review-process/

The reviewer guide asks whether the work is clear, technically correct, experimentally rigorous, reproducible, novel, well placed in the literature, supported by its evidence, significant, and valuable to the ICLR community. State-of-the-art results are not required.

## Required Review Fields

Write:

1. summary;
2. Soundness score and justification;
3. Presentation score and justification;
4. Contribution score and justification;
5. strengths;
6. weaknesses;
7. questions for the authors;
8. ethics flag and explanation when applicable;
9. overall Rating and one or two decision-driving reasons;
10. Confidence;
11. LLM-use disclosure for the internal automated review.

## Overall Rating

Use exactly one integer from this set:

- **10 — strong accept:** should be highlighted at the conference as a spotlight or oral.
- **8 — accept:** good paper suitable for a poster.
- **6 — marginally above the acceptance threshold:** reasons to accept narrowly outweigh reasons to reject; would not mind rejection.
- **4 — marginally below the acceptance threshold:** reasons to reject narrowly outweigh reasons to accept; would not mind acceptance.
- **2 — reject:** not good enough for ICLR in its current form.

Do not use 0, odd integers, decimals, prose-only scores, or interpolated values. Do not average the three dimension scores to obtain the rating.

## Official Dimensions: 1--4

### Soundness

- **4 — excellent:** central claims are technically correct and unusually well justified; empirical or theoretical evidence is decisive for the stated scope.
- **3 — good:** central claims are sound with only minor, non-decision-changing gaps.
- **2 — fair:** plausible core, but one or more substantive gaps materially reduce confidence.
- **1 — poor:** central logic, method, or evidence is invalid or cannot support the claims.

### Presentation

- **4 — excellent:** exceptionally clear, well structured, precise, and easy to verify.
- **3 — good:** clear and readable with localized presentation problems.
- **2 — fair:** understandable, but organization, definitions, figures, or writing materially obstruct evaluation.
- **1 — poor:** difficult or impossible to evaluate reliably because of presentation.

### Contribution

- **4 — excellent:** substantial new knowledge or capability with compelling importance to the ICLR community.
- **3 — good:** clear, useful, and sufficiently distinct contribution with credible significance.
- **2 — fair:** narrow, incremental, weakly positioned, or of uncertain significance.
- **1 — poor:** no identifiable new knowledge or decision-relevant contribution.

## Confidence: 1--5

- **5:** absolutely certain; expert in the area and checked central technical details carefully.
- **4:** confident but not absolutely certain; remaining uncertainty is unlikely to change the decision.
- **3:** fairly confident; some misunderstanding or missing related-work knowledge could change the assessment.
- **2:** willing to defend the assessment, but substantial uncertainty remains about central parts or related work.
- **1:** educated guess; expertise or evidence access is too limited for a dependable decision.

## Decision Procedure

Before assigning a Rating, answer the four questions emphasized by ICLR:

1. What specific problem does the paper tackle?
2. Is the approach motivated and well placed in the literature?
3. Do the results correctly and rigorously support the claims?
4. Does the work contribute new, relevant, impactful knowledge or practical value?

Then identify:

- the strongest verified contribution;
- the most severe unresolved issue;
- the evidence ceiling without new experiments;
- whether the paper is decision-ready at ICLR's page and anonymity constraints.

## Internal Completion Target

The autonomous loop may stop as quality-passed only when:

- five independent reviews exist;
- panel median Rating is at least 8;
- lower quartile Rating is at least 6;
- at least four reviewers rate 6 or higher;
- no supported Rating 2 remains;
- median Soundness, Presentation, and Contribution are each at least 3;
- no unresolved critical or major technical/evidence/provenance issue remains;
- the independent meta-review Rating is at least 6;
- integrity, build, anonymization, page-limit, citation, and provenance gates pass.

These are conservative internal gates. They do not imply any acceptance probability.

## ICLR 2026 Format Checks

- Use the official ICLR 2026 template.
- Keep initial-submission main text to at most 9 pages; references and appendices are excluded as specified by the Author Guide.
- Preserve double-blind anonymity.
- Include a concise reproducibility statement before the references.
- Evaluate whether an ethics statement is warranted.
- Record significant LLM use for the required submission disclosure.

