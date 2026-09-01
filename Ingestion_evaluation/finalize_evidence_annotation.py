"""Merge reviewed sufficiency decisions into the all-60 retrieval bundle."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


DECISIONS = {
    "Q1": ("pass", "Retrieved HDB01 text explicitly lists eligible core family nuclei and notes that eligibility depends on household conditions."),
    "Q2": ("pass", "Retrieved HDB grant text explicitly describes a first-timer citizen buying a resale flat on their own, which supports the singles route despite HDB02 not appearing."),
    "Q3": ("pass", "Rank 1 states the Singles Grant amounts by resale-flat size and retrieved text also identifies EHG and PHG as conditional additions."),
    "Q4": ("pass", "Retrieved HDB06 and HDB05 passages identify the family resale grant route, EHG Families, and PHG Families."),
    "Q5": ("fail", "The retrieved Step-Up passages say second-timer families may apply but omit the defining current-housing and next-flat conditions needed for a complete answer."),
    "Q6": ("fail", "HDB08 is retrieved, but its selected chunk does not contain the $16,000, $24,000, and $8,000 income ceilings requested."),
    "Q7": ("pass", "Retrieved HDB09 text lists upfront cash/CPF payments for new and resale flats, including deposits, fees, and cash above valuation."),
    "Q8": ("pass", "Retrieved HDB10 guidance directs buyers to the HDB Flat Portal and its new-flat listings/current sales resources."),
    "Q9": ("fail", "The results contain general resale guidance and OTP details but not the complete sequence of main resale-purchase stages."),
    "Q10": ("fail", "An HDB12 chunk is retrieved, but the selected passage does not state the 21-calendar-day Option Period."),
    "Q11": ("pass", "The CPF01 chunk explicitly names OA, MA, SA and RA and states the age-55 RA creation and SA closure."),
    "Q12": ("fail", "The only retrieved CPF02 passage is a generic tools sentence and does not establish CPF's housing, healthcare, and retirement purposes."),
    "Q13": ("pass", "Multiple retrieved passages explicitly identify Ordinary Account savings as usable for eligible home purchases and housing payments."),
    "Q14": ("pass", "CPF06 text states that CPF principal used plus accrued interest must generally be refunded on sale or transfer."),
    "Q15": ("pass", "CPF08 explicitly defines CPF LIFE as a national longevity insurance annuity providing monthly payouts for life."),
    "Q16": ("pass", "Retrieved HDB03/HDB04 passages list Singles Grant, EHG and PHG and explicitly state the resale-grant prerequisite for EHG."),
    "Q17": ("pass", "HDB05 explicitly states that the family resale grant must be qualified for before EHG and gives EHG's additional means-tested conditions."),
    "Q18": ("fail", "Retrieved text identifies Step-Up but omits the specific current-housing and eligible next-flat circumstances central to the question."),
    "Q19": ("fail", "The retrieved loan passage mentions HFE and loan assessment but omits the combined grants, available-funds, and budgeting relationship."),
    "Q20": ("fail", "OTP exercise and resale submission evidence is present, but the retrieved set does not fully support the financing/request-for-value steps across the requested sequence."),
    "Q21": ("pass", "Retrieved CPF03 and CPF06 text connects OA housing use with later refund of the amount used plus accrued interest to restore retirement savings."),
    "Q22": ("pass", "CPF01 directly states that housing use reduces retirement resources, and retrieved retirement text supports the role of CPF savings in later payouts."),
    "Q23": ("pass", "CPF07/CPF08 passages establish CPF LIFE's lifelong monthly payouts and its connection to retirement savings and Retirement Account funding."),
    "Q24": ("pass", "CPF10 provides the base account rates and the complete age-55-plus extra-interest tiers and OA cap."),
    "Q25": ("fail", "HDB budgeting evidence is strong, but the retrieved top five omit the complementary CPF guidance on OA usage limits and retirement preservation."),
    "Q26": ("pass", "The retrieved HDB04 passage states the Singles Grant prerequisite and additional EHG conditions, with detailed eligibility text also present."),
    "Q27": ("pass", "HDB05 explicitly states that the family resale grant must be established first and supplies additional EHG conditions."),
    "Q28": ("fail", "The results state that second-timer families may apply but do not retrieve the scheme's specific current-housing and next-flat requirements."),
    "Q29": ("pass", "HDB08 explicitly states the $8,000 income ceiling for singles under the SSC Scheme."),
    "Q30": ("pass", "HDB08 explicitly states that no HDB loan is available for short-lease 2-room Flexi flats and identifies cash/CPF OA payment."),
    "Q31": ("pass", "HDB12 states that no action is needed, the OTP expires, and the Option Fee is forfeited if the buyer does not proceed."),
    "Q32": ("fail", "The retrieved text supports refund of principal plus accrued interest but omits the gold answer's housing-grant refund component."),
    "Q33": ("pass", "CPF06 shows that refund outcomes depend on sale circumstances, exceptions and personalised dashboard information, so an automatic cash-top-up claim is unsupported."),
    "Q34": ("pass", "CPF08 explicitly states up to 7% higher payouts per year of deferral, up to age 70."),
    "Q35": ("pass", "CPF10 clearly limits the displayed rate to July-September 2026, providing sufficient evidence not to carry it into another period."),
    "Q36": ("pass", "The results show multiple conditional grant routes, amounts and eligibility factors and direct the user to HFE, supporting a qualified non-personalised response."),
    "Q37": ("pass", "HDB08 explicitly distinguishes income eligibility from credit assessment and lists the other factors determining approval and amount."),
    "Q38": ("pass", "Retrieved HDB/CPF passages show that downpayment funds alone do not establish affordability and identify wider financing and retirement considerations."),
    "Q39": ("pass", "CPF06 explains the general formula, personal exceptions, and the Home ownership dashboard needed for the exact amount."),
    "Q40": ("pass", "CPF08 shows that plan, savings/premium and payout start or deferral affect payouts, supporting refusal to infer an exact amount from age alone."),
    "Q41": ("pass", "HDB04 states that the resale grant is only a prerequisite and retrieves separate EHG income and household assessment conditions."),
    "Q42": ("pass", "The Step-Up passages use conditional language and include additional household/property requirements, supporting correction of the guarantee premise."),
    "Q43": ("pass", "HDB09 identifies payment types, financing variables, calculators and a customised payment plan, supporting a qualified response rather than an exact calculation."),
    "Q44": ("pass", "HDB08 lists financial and credit-assessment factors and states that HFE provides the assessed amount, so the exact loan cannot be inferred from income alone."),
    "Q45": ("pass", "Retrieved HDB material separates flat eligibility from conditional grant eligibility and repeatedly states that individual grant conditions apply."),
    "Q46": ("fail", "The retrieved source only establishes a 2026 quarter and cannot establish an OA rate for 2035."),
    "Q47": ("fail", "Current HDB ceilings and historical examples cannot establish the policy value that will apply in 2030."),
    "Q48": ("fail", "The retrieved pages provide current/historical resale-price resources but no reliable forecast of next year's highest-priced town."),
    "Q49": ("fail", "Public informational pages contain no authenticated personal OA balance."),
    "Q50": ("fail", "The retrieved budgeting page directs users to compare financial-institution packages but does not establish a current personalised best mortgage rate."),
    "Q51": ("fail", "The retrieved material contains no reliable five-year forecast for the user's property value."),
    "Q52": ("fail", "Current housing-policy pages cannot establish announcements at a future Budget."),
    "Q53": ("pass", "HDB02 explicitly states the general age-35 singles requirement and notes further eligibility conditions, correcting the false premise."),
    "Q54": ("pass", "CPF03 explicitly identifies OA—not SA—as the account used for eligible housing purchases and downpayments."),
    "Q55": ("pass", "CPF06 explicitly states that both CPF principal and accrued interest generally form the housing refund."),
    "Q56": ("pass", "HDB12 states the 21-day expiry and consequences of expiry, with no automatic extension."),
    "Q57": ("pass", "CPF10 explicitly states a 2.5% OA rate for 1 July through 30 September 2026."),
    "Q58": ("fail", "The retrieved rate applies only through September 2026 and cannot establish the October-December 2026 rate."),
    "Q59": ("pass", "CPF10 explicitly states the applicable period as 1 July through 30 September 2026."),
    "Q60": ("fail", "A current-quarter CPF rate cannot establish next year's OA rate."),
}

# These previous binary failures contain useful answer evidence but omit at least one
# material component. Remaining failures have no usable evidence for the requested
# answer (including future, forecast, and private-account questions).
PARTIAL_IDS = {"Q5", "Q9", "Q18", "Q19", "Q20", "Q25", "Q28", "Q32"}


def main() -> None:
    """Merge reviewed decisions into the annotation bundle and export reports."""
    root = Path(__file__).resolve().parent
    bundle_path = root / "data/experiments/evidence_sufficiency/retrieval_bundle_all_60.jsonl"
    records = [json.loads(line) for line in bundle_path.read_text(encoding="utf-8").splitlines()]
    bundle_ids = {record["question_id"] for record in records}
    if bundle_ids != set(DECISIONS):
        raise ValueError(f"Decision IDs do not match bundle: {sorted(bundle_ids ^ set(DECISIONS))}")

    annotated = []
    flat_rows = []
    for record in records:
        binary_label, rationale = DECISIONS[record["question_id"]]
        if binary_label == "pass":
            label = "SUFFICIENT"
            behavior = "answer_or_qualify_using_retrieved_evidence"
        elif record["question_id"] in PARTIAL_IDS:
            label = "PARTIAL"
            behavior = "give_only_supported_partial_answer_and_explicitly_abstain_on_missing_elements"
        else:
            label = "INSUFFICIENT"
            behavior = "abstain_due_to_insufficient_retrieved_evidence"
        record["evidence_label"] = label
        record["evidence_sufficiency"] = label
        record["correct_system_behavior"] = behavior
        record["annotation_rationale"] = rationale
        record["annotation_status"] = "ai_reviewed_draft_pending_human_validation"
        annotated.append(record)
        row = {key: value for key, value in record.items() if key != "retrieved_chunks"}
        row["gold_source_ids"] = ", ".join(record["gold_source_ids"])
        for chunk in record["retrieved_chunks"]:
            rank = chunk["rank"]
            for field in ("chunk_id", "source_id", "title", "heading_path", "score", "text"):
                row[f"rank_{rank}_{field}"] = chunk[field]
        flat_rows.append(row)

    output_dir = bundle_path.parent
    jsonl_path = output_dir / "evidence_sufficiency_annotations_all_60.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in annotated:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_path = output_dir / "evidence_sufficiency_annotations_all_60.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {
        "total": len(annotated),
        "labels": dict(Counter(record["evidence_label"] for record in annotated)),
        "annotation_status": "ai_reviewed_draft_pending_human_validation",
        "label_definitions": {
            "SUFFICIENT": "Top-five text fully supports a safe answer or qualified correction to the benchmark question.",
            "PARTIAL": "Top-five text contains useful evidence but omits at least one material fact or required component.",
            "INSUFFICIENT": "Top-five text provides no usable basis for the requested answer, or the answer is future, forecast, or private-account information unavailable in the corpus."
        },
        "behavior_policy": {
            "SUFFICIENT": "Answer using only retrieved evidence.",
            "PARTIAL": "Provide only the supported portion, disclose the evidence gap, and abstain from missing claims.",
            "INSUFFICIENT": "Abstain."
        }
    }
    (output_dir / "annotation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
