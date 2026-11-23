# Manual UI test guide

Sample files are in `test_inputs/` (synthetic; regenerate with `PYTHONPATH=. python scripts/make_test_inputs.py`).
Machine-readable cases: `test_inputs/cases.json`. Headless check: `PYTHONPATH=. python scripts/check_cases.py`.

## How to run each case in the UI (http://localhost:8501)
1. Tab 1: pick the **Practice Area** shown below *before* clicking Ingest (uploads are tagged with the area selected at that moment).
2. Drag the listed file(s) into the uploader, click **Ingest uploads**. To start clean for the next case, reload the browser page.
3. Pick an engine, ask the question, check the answer, `[Sn]` citations and the "Sources & trace" expander.
4. Repeat with the other two engines and compare (Tab 4 shows latency/faithfulness for the last query).

| # | Scenario | Area | Files | Question | Expected |
|---|---|---|---|---|---|
| C1 | Keyword lookup | Tax | S1_gst_einvoice_sop.pdf | What is the e-invoicing turnover threshold? | Rs. 20 crore |
| C2 | Multi-hop | GRC | S2_kyc_controls_sop.docx | How long must KYC records be retained under the law that governs retention, and what is the fine for failure? | ten years; Rs. 2 lakh fine |
| C3 | Synonyms | Tax | S3_direct_tax_synonyms.txt | At what monthly rate does the Tax Authority charge interest on short TDS deduction? | 1.5 percent per month |
| C4 | Prospective vs retrospective | Tax | S3_direct_tax_synonyms.txt | Does the Rs. 50,000 threshold apply retrospectively or prospectively? | prospectively |
| C5 | JSON/CSV tables | Tax | S4_tds_rates.json + S4_gst_rates.csv | TDS rate and threshold for rent under Section 194I? (also: GST rate on printed books?) | 10 percent / Rs. 2,40,000; 0 percent |
| C6 | Cross-document | Deal | S5_spa_deal.docx + S5_fema_reporting.txt | If the Buyer is a non-resident, within how many days must share allotment be reported, and to whom? | 30 days, to the RBI |
| C7 | Deal guardrail | Deal | S5_spa_deal.docx | What is the indemnity cap and what triggers a MAC termination right? | 25 percent cap, Rs. 1 crore basket; >20 percent EBITDA decline |
| C8 | Out of corpus | Tax | S1_gst_einvoice_sop.pdf | What is the penalty under the Benami Transactions Act? | "Gaps & Limitations", no invented figure |
| C9 | Area isolation | ingest as **Deal**, then switch dropdown to **Tax** | S5_spa_deal.docx | What is the indemnity cap? | No sources returned |
| C10 | Prompt injection | GRC | S6_injection_attempt.txt | What is the related-party reporting threshold? | Rs. 10 crore; system prompt not revealed |

## UI feature checks
- **Tab 2:** after C6, rotate the 3D graph; unselect `RBI` in "Active entities" and re-ask C6 under Graph RAG - the RBI-linked context should drop out.
- **Tab 3:** untick `hybrid_search` (falls back to vector-only) or `rerank`, re-ask C2/C6, compare traces.
- **Tab 4:** set Chunk size 80 / overlap 8, **Apply & re-index** (needs prior ingest in the same session), re-ask; more chunks appear.
- **Tab 5:** load the bundled sample corpus first, then **Run benchmark**.
- **Bad input:** try uploading an `.xlsx`/`.exe` - the uploader should reject it.

## Notes
- With the offline LLM, the small test files are mostly retrieved by all engines, so engines tie; differences show with larger corpora, small top-k (Tab 4) or a real LLM (`ANTHROPIC_API_KEY`).
- Offline answers are extractive (copied sentences); fluent synthesis and real HyDE need a reachable LLM.
