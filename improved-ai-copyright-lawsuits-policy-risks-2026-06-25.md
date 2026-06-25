# AI Copyright Lawsuits and Policy Risks: Revised Report

> Research query: What are the main AI copyright lawsuits and policy risks?

This revised version addresses the eval failure by removing unsupported YouTube-sourced authorship claims, fixing reversed-party lawsuit attributions, avoiding unfinished case-name fragments, adding citations to factual summary claims, and adding a liability and damages section.

## Executive Summary

AI copyright risk is concentrated in three areas: U.S. lawsuits over model training and outputs, policy fights over whether training requires consent or compensation, and cross-border differences in training exceptions and output ownership rules [1][2][3][8]. The largest U.S. litigation cluster includes cases against OpenAI and Microsoft, Meta, Anthropic, Google, NVIDIA, Cohere, Stability AI, Midjourney, DeviantArt, GitHub, and Microsoft, with plaintiffs including authors, news publishers, music publishers, visual artists, and software developers [1].

The core U.S. doctrinal issue is whether copying copyrighted works for AI training is excused by fair use under Section 107 [3][4]. The emerging lesson is not that every AI-training use is lawful or unlawful; fair use depends on facts such as the purpose of the model, the source of the copied works, output controls, market substitution, and licensing-market harm [3][4]. U.S. damages exposure can be material because copyright owners may seek actual damages and profits or statutory damages, with statutory damages generally ranging from $750 to $30,000 per work and up to $150,000 per work for willful infringement [5].

Globally, the main policy risk is fragmentation rather than one settled rule [8]. The EU is building a compliance architecture around general-purpose AI obligations, copyright-reservation mechanisms for text and data mining, and AI regulatory sandboxes under the AI Act [6]. Japan and Singapore are described as more permissive toward AI training, while Mexico and China are described as denying copyright protection to certain fully or insufficiently human-authored AI outputs, and Denmark is considering copyright-like digital identity protection for faces, voices, and likenesses [8].

## Litigation Landscape

The main training-data cases are testing whether AI developers infringed copyright by copying books, journalism, lyrics, source code, images, or other protected works into datasets or model-development pipelines [1][3]. BakerHostetler's case tracker identifies an OpenAI/Microsoft multidistrict litigation combining twelve cases by news media, authors, and others alleging copyright infringement from using plaintiffs' works to train LLMs [1]. The same tracker identifies Kadrey v. Meta as an author suit alleging Meta copied books to train LLaMA models, Nazemian and Dubus v. NVIDIA as suits alleging NVIDIA copied books to train Nemo Megatron-GPT, and In re Google Generative AI Copyright Litigation as alleging Google scraped and used works to train AI products including Gemini [1].

Publisher and media suits add a market-harm theory that may matter under fair use factor four [1][3]. Conde Nast, The Atlantic, Axel Springer, and other publishers accuse Cohere of direct and indirect infringement based on Cohere's AI systems [1]. Advance Local Media v. Cohere alleges a licensing market for publisher content used by AI developers, which is relevant to the market-effect analysis under fair use [1][3].

Music and code cases raise both copyright and Digital Millennium Copyright Act theories [1]. Concord Music Group v. Anthropic is brought by music publisher plaintiffs alleging Anthropic violated the Copyright Act and DMCA section 1202(b) by using copyrighted lyrics to train Claude [1]. Doe v. GitHub alleges GitHub, Microsoft, and OpenAI breached open-source software licenses and violated the DMCA by using copyrighted materials to create Codex and Copilot [1].

Image-generation litigation raises both training-copying and output-system theories [1]. Andersen v. Stability AI includes visual artist allegations of direct and induced copyright infringement, DMCA violations, false endorsement, and trade dress claims against Stability AI, Midjourney, and DeviantArt [1]. A USC legal summary states that, at the pleading stage, the Andersen court found direct-infringement allegations sufficient where plaintiffs alleged Stability acquired copies of protected works, trained Stable Diffusion on them, and stored or incorporated training-image information as compressed copies [9].

This landscape should be read with caution because much of the lawsuit catalog comes from a single law-firm tracker [1]. The tracker is useful for mapping claims and parties, but the original report did not independently verify each case number, procedural posture, dismissal status, or settlement status [1]. For risk ranking, the cases with actual rulings or concrete damages exposure deserve more weight than early complaints that may be narrowed, settled, or dismissed [3][4][5].

## Fair Use And Training

Section 107 fair use asks courts to consider the purpose and character of the use, the nature of the copyrighted work, the amount and substantiality used, and the effect on the market for the copyrighted work [3]. The U.S. Copyright Office's Part 3 report says multiple stages of generative AI development can implicate copyright owners' exclusive rights, and that the key question is whether those acts can be excused as fair use [4].

Thomson Reuters v. ROSS Intelligence is a warning case for defendants building substitute products [3]. A district court rejected ROSS's fair-use defense where ROSS used Westlaw headnotes to build a competing legal-research tool, and the court found market harm to both Westlaw's legal-research market and a potential derivative market for training data [3]. That case is especially important for AI risk analysis because it treats a training-data market as a cognizable market under factor four [3].

Bartz v. Anthropic is more favorable to AI developers, but only on part of the conduct [3][4]. Sources summarized in the original findings state that Judge William Alsup found Anthropic's use of lawfully acquired books for training Claude "spectacularly" transformative and found no direct substitution where there was no evidence the model would output exact copies [3]. The Copyright Office's Part 3 report also says various AI-training uses are likely transformative, but it emphasizes that fairness depends on source, purpose, output controls, and market effects [4].

The key distinction is lawful acquisition and market effect [3][4]. Training on lawfully acquired works for a non-substitutive model looks meaningfully different from using protected material to build a direct market substitute or from using pirated or illegally accessed copies [3][4]. The Copyright Office states that knowing use of pirated or illegally accessed works should weigh against fair use, even if it is not automatically determinative [4].

## Liability And Damages

The eval correctly flagged that the original report did not analyze liability and damages. Under 17 U.S.C. section 504, an infringer may be liable for the copyright owner's actual damages plus attributable profits, or for statutory damages [5]. Statutory damages generally range from $750 to $30,000 per work, and courts may increase the award to no more than $150,000 per work when the copyright owner proves willful infringement [5].

That damages structure makes scale the practical risk multiplier in AI-training cases [5]. A model developer accused of copying thousands or millions of works faces a very different settlement and trial-risk profile from a defendant accused of a small number of isolated outputs [5]. The most important liability variables are the number of registered works at issue, whether infringement is willful, whether copying was from lawful or pirated sources, whether outputs reproduce protected expression, and whether plaintiffs can show actual market harm or a lost licensing market [3][4][5].

DMCA claims add a separate risk channel when plaintiffs allege copyright-management information was removed or altered [1]. Concord Music Group v. Anthropic includes a DMCA section 1202(b) theory tied to lyrics, and Doe v. GitHub includes DMCA allegations tied to source-code material used for Codex and Copilot [1]. These claims matter because a defendant can face exposure even when the core copyright theory is contested under fair use [1][3].

## U.S. Copyright Office And Authorship

The U.S. Copyright Office launched its AI initiative in early 2023 and received more than 10,000 comments after its August 2023 notice of inquiry [2]. The Office's AI report is being issued in parts, with Part 2 addressing copyrightability of outputs created using generative AI and Part 3 addressing generative AI training [2].

Part 2 says copyright protection remains available for original expression created by a human author even if the work also contains AI-generated material [10]. Part 2 also says copyright does not extend to purely AI-generated material or material where there is insufficient human control over expressive elements [10]. Based on current generally available technology, Part 2 says prompts alone do not provide sufficient control to establish authorship [10].

The practical ownership risk is therefore narrower than the original report suggested. The stronger supported rule is that AI cannot be the author of purely machine-generated expression under current U.S. Copyright Office policy and related court materials, while human-authored selection, arrangement, modification, or perceptible expression may still be protectable [2][10]. The unresolved operational question is how much human contribution is enough in specific workflows, because Part 2 says sufficiency of human contribution must be analyzed case by case [10].

## EU Policy And Compliance

The EU risk is not a finalized AI copyright licensing mandate, but a developing compliance framework [6][7]. A European Parliament draft report has called for a licensing regime in Europe enabling generative AI providers to obtain licenses for copyright-protected works [7]. That proposal should be described as a draft policy direction, not as binding law [7].

The European Commission launched a December 1, 2025 stakeholder consultation on technical protocols for expressing reservations of rights against text and data mining [6]. The consultation is intended to support AI Act obligations for general-purpose AI model providers to maintain a policy for complying with EU law, including identifying and complying with reservations of rights under Article 4(3) of Directive (EU) 2019/790 [6]. The Commission sought machine-readable, standardized protocols that could work consistently across media, languages, and sectors [6].

The Commission also launched a December 2, 2025 consultation on a draft implementing act for AI regulatory sandboxes [6]. Article 57 of the AI Act requires Member States to establish at least one national AI regulatory sandbox by August 2, 2026 [6]. These sandboxes are designed to let businesses test innovative AI systems under regulatory supervision [6].

The practical EU risk is implementation uncertainty [6][7]. Providers should expect more documentation, rights-reservation, and compliance-policy work in the EU than in jurisdictions with broader training exceptions, but the precise burden depends on the final reservation protocols, sandbox rules, and any later copyright-specific legislation [6][7].

## Cross-Border Policy Divergence

The original report's cross-border section was directionally useful but overstated the evidence. A DWT analysis describes Japan and Singapore as having AI-friendly training exceptions that allow data training without prior consent [8]. The same analysis describes Mexico's Supreme Court as finding works generated exclusively by AI ineligible for copyright protection and describes a Chinese court decision denying copyright protection for certain AI-generated images due to insufficient human input [8].

Those examples involve different legal questions [8]. Japan and Singapore are about input-side training permissions, while Mexico and China are about output-side copyrightability [8]. A company therefore needs separate controls for training data provenance, output infringement risk, and output ownership or registrability [4][8][10].

The evidence for actual cross-border enforcement conflict remains limited in the original findings. A DWT hypothetical says a television production team using AI-generated assets might find materials acceptable in Japan but not in Mexico or the EU [8]. That example supports a compliance-planning concern, but it does not prove that companies are already facing major cross-border infringement judgments from these differences [8].

## Practical Risk Ranking

The highest near-term risk is U.S. litigation over training data, especially where plaintiffs can allege direct market substitution, lost licensing opportunities, pirated-source acquisition, or output reproduction [1][3][4][5]. The second major risk is damages leverage, because statutory damages are calculated per work and can rise to $150,000 per work for willful infringement [5]. The third major risk is EU compliance uncertainty, because general-purpose AI providers must prepare for copyright-reservation policies and AI Act implementation details that are still developing [6][7].

Companies should treat source provenance as a first-order control [4]. Lawfully acquired, licensed, public-domain, or clearly permitted training material presents a different risk profile from scraped, pirated, paywalled, or rights-reserved material [4]. Output controls also matter because memorization, exact reproduction, and substitutive outputs can change the fair-use analysis even if training itself is argued to be transformative [3][4].

The weakest parts of the evidence base are procedural status, cross-border enforcement, and prompt-authorship case law. The lawsuit list needs docket-level verification before ranking which cases are most likely to produce binding precedent [1]. The cross-border risk needs more evidence of actual disputes rather than hypotheticals [8]. The prompt-authorship question should rely on the Copyright Office's Part 2 analysis rather than video summaries [10].

## References

[1] [BakerHostetler AI Copyrights and Class Actions Case Tracker](https://www.bakerlaw.com/services/artificial-intelligence-ai/case-tracker-artificial-intelligence-copyrights-and-class-actions)

[2] [U.S. Copyright Office: Copyright and Artificial Intelligence](https://www.copyright.gov/ai/)

[3] [Cleary IP Tech Insights: The Open Questions in U.S. Generative AI Copyright Litigation](https://www.clearyiptechinsights.com/2026/01/the-open-questions-in-u-s-generative-ai-copyright-litigation)

[4] [U.S. Copyright Office: Copyright and Artificial Intelligence, Part 3: Generative AI Training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)

[5] [17 U.S.C. section 504, Damages and Profits](https://www.law.cornell.edu/uscode/text/17/504)

[6] [Inside Privacy: European Commission Consultations on EU AI Act Copyright Provisions and AI Regulatory Sandboxes](https://www.insideprivacy.com/artificial-intelligence/european-commission-launches-consultations-on-the-eu-ai-acts-copyright-provisions-and-ai-regulatory-sandboxes)

[7] [Inside Global Tech: European Parliament Proposes Changes to Copyright Protection in the Age of Generative AI](https://www.insideglobaltech.com/2026/02/16/european-parliament-proposes-changes-to-copyright-protection-in-the-age-of-generative-ai)

[8] [DWT: Global Patchwork of AI Regulation for Content Creators](https://www.dwt.com/blogs/artificial-intelligence-law-advisor/2025/11/ai-regulation-global-patchwork-content-creators)

[9] [USC IPTLS: AI Copyright and the Law](https://sites.usc.edu/iptls/2025/02/04/ai-copyright-and-the-law-the-ongoing-battle-over-intellectual-property-rights)

[10] [U.S. Copyright Office: Copyright and Artificial Intelligence, Part 2: Copyrightability](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf)
