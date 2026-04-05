# Example Data Creation Guide

## Purpose

This directory contains sample **service agreement** documents and their expected structured extractions. These examples serve as review/reference material for the meta-learning document extraction system — they are **not** consumed by the system directly.

## Directory Structure

```
examples/
└── service_agreements/
    ├── inputs/          # Raw text files (sa_001.txt, sa_002.txt, ...)
    ├── expected/        # Corresponding JSON extractions (sa_001.json, sa_002.json, ...)
    └── DATA_GUIDE.md    # This file
```

## File Naming Convention

- **Inputs**: `sa_NNN.txt` where NNN is a zero-padded 3-digit sequential number (001–999)
- **Expected**: `sa_NNN.json` matching the same number

---

## How to Create a New Example

### Step 1: Write the Input Document

Create a realistic service agreement text file in `inputs/sa_NNN.txt`. Follow these guidelines:

**Length**: 250–800 lines of plain text. The documents should be long enough to force the extraction system to search through significant noise to find relevant fields. Short, clean documents will not adequately test retrieval.

**Required Sections** (use natural language, vary the headings):
1. **Header** — agreement number/ID and effective date
2. **Parties** — service provider and client with legal names, entity types, jurisdictions, addresses
3. **Scope of Services** — description of what will be performed
4. **Term / Duration** — start date, end date or duration, renewal terms
5. **Compensation** — fee structure (monthly retainer, hourly, fixed project, milestone-based)
6. **Deliverables / Milestones** — concrete outputs with dates or timeframes
7. **Intellectual Property** — who owns what, license grants
8. **Confidentiality / NDA** — duration of obligations
9. **Termination** — notice periods, cause vs. convenience
10. **Liability** — caps, exclusions of consequential damages
11. **Governing Law** — jurisdiction and country

**Vary these intentionally across examples**:
- Entity types (LLC, corporation, sole proprietorship, partnership, LLP, PLLC, REIT, JPA, nonprofit)
- Fee structures (hourly, fixed, retainer, milestone-based, percentage, per-unit, revenue sharing, in-kind)
- Jurisdictions (different US states, international — UK, etc.)
- Term lengths (6 months to 60+ months)
- Auto-renewal vs. fixed term
- IP ownership (provider-owned, client-owned, split, joint, copyright retention)
- Liability caps (dollar amount, months of fees, total project value, per-occurrence, aggregate, uncapped)
- Dispute resolution methods (litigation, arbitration, mediation, or absent entirely)
- Insurance requirements (some agreements, not others — varies by industry)
- Non-solicitation / non-compete / anti-poaching clauses
- Warranty periods
- Number of parties (two-party standard, multi-party for consortiums)

**Include edge cases deliberately** (sprinkle these across the set, not in every file):
- Missing end date (open-ended with notice)
- No renewal terms mentioned
- No IP clause (implicit ownership)
- No liability cap (unlimited liability)
- No dispute resolution clause
- No insurance requirements
- Multiple signatories with witnesses
- Amendments or addenda referenced
- Exhibits or schedules referenced but not included ("attached but not included")
- Currency other than USD
- Partial information (e.g., no jurisdiction for one party)
- Ambiguous terms that a reasonable extractor might interpret differently
- Redacted or incomplete information (e.g., `RDC-2024-[REDACTED]`)
- Effective date buried in footnote or signature block rather than header
- Contingent fees (e.g., Phase II only if Phase I findings warrant)
- Mixed cash and in-kind compensation
- Per-phase or per-matter termination rights
- KPI-based termination triggers
- Bankruptcy or insolvency termination triggers
- Scientific/technical infeasibility termination triggers
- Tiered pricing by category (language tier, researcher level, complexity)
- Rush surcharges and overtime rates
- Minimum engagement levels (monthly minimums, minimum billable hours)
- Success bonuses or penalty clauses

**Realism tips**:
- Use plausible company names and addresses
- Use realistic dollar amounts for the industry
- Reference real-world regulation (HIPAA, GDPR, SOX, NEPA, FDA, FCC, PCAOB, OSHA, ADA, FMLA, PCI DSS, FCA, PSD2) where appropriate
- Include typical contract boilerplate language
- Vary signing dates vs. effective dates (signing can precede or follow effective date)
- Use realistic signatory titles (CEO, COO, VP, Managing Partner, CTO, CFO, SVP, Director, etc.)

### Data Dilution (Required)

Every input document **must** contain significant amounts of irrelevant content that buries the extraction-relevant fields. This is not optional — it is the core challenge for the retrieval and extraction system.

**Target signal-to-noise ratio**: Extraction-relevant data should account for roughly 15–25% of the document. The remaining 75–85% is noise.

**Dilution techniques** (use most of these in every document):

1. **Preamble / Recitals**: 4–10 WHEREAS clauses providing industry context, market background, and strategic rationale. These should reference real regulations, industry trends, and business context — making them plausible and not obviously filler.

2. **Definitions section**: 15–30 defined terms relevant to the domain. These definitions are useful for understanding the contract but do not contain extraction-relevant data themselves. Use domain-specific terminology (e.g., HIPAA, NEPA, GPON, LEED, SOC 2, ASTM, NIST, COSO, OWASP, HACCP, BIM, etc.).

3. **Operational detail sections**: 4–8 sections describing HOW the services are performed (methodology, procedures, standards, tools, processes). These are realistic but irrelevant to the extraction schema. Examples:
   - Quality assurance procedures and acceptance criteria
   - Health and safety requirements (OSHA, PPE, emergency response)
   - Technology stack descriptions (platforms, tools, integrations)
   - Personnel qualification requirements (certifications, background checks, training)
   - Regulatory compliance procedures (specific to the industry)
   - Reporting formats, cadence, and distribution
   - Communication protocols and escalation matrices
   - Change management and version control procedures
   - Data handling, privacy, and security procedures
   - Vendor/subcontractor management procedures

4. **General boilerplate**: Entire agreement, amendments, assignment, severability, notices, counterparts, relationship of parties, force majeure, survival, no third-party beneficiaries — standard legal provisions that exist in every contract but rarely contain extraction-relevant data.

5. **Exhibit references**: 2–4 exhibits described as "attached but not included." These create false leads — an extraction system might look for data in referenced exhibits that don't exist.

6. **Witness signature blocks**: Additional signature lines for witnesses or notaries, adding visual noise around the actual signatures.

7. **Distribute key data**: Do NOT cluster extraction-relevant fields together. Spread them across the document — put the agreement number at the top, party info in the preamble, compensation deep in a middle section, liability near the end, signatures at the very bottom. The relevant fields should not be findable with a simple scan.

8. **Industry-specific noise**: Include sections that are highly relevant to the specific industry but contain no schema-relevant data:
   - Healthcare: HIPAA breach notification, EHR system requirements, clinical workflows
   - Construction: BIM coordination, LEED certification, safety incident reporting
   - Cybersecurity: NIST CSF, MITRE ATT&CK, penetration testing rules of engagement
   - Food/beverage: HACCP plans, ABC licensing, allergen protocols
   - Environmental: ASTM standards, wetlands delineation methodology, species survey protocols
   - Financial: PCAOB inspection readiness, SOX 404 scoping, audit committee communication
   - Legal: privilege log format, billing standards, conflicts screening procedures

### Step 2: Write the Expected Extraction

Create a corresponding JSON file in `expected/sa_NNN.json`. Follow the schema below.

#### Extraction Schema (Service Agreements)

The base schema below covers all service agreements. Domain-specific fields may be added as needed (see the "Schema Extensions" section).

```json
{
  "agreement_number": "string | null — identifier from the document",
  "effective_date": "string | null — ISO 8601 date (YYYY-MM-DD)",

  "service_provider": {
    "name": "string",
    "type": "string | null — corporation, LLC, LLP, PLLC, partnership, sole proprietorship, nonprofit, REIT, JPA",
    "jurisdiction": "string | null — state/country of incorporation/organization",
    "address": "string | null",
    "signatory": "string | null",
    "signatory_title": "string | null",
    "signatory_date": "string | null — ISO 8601 date"
  },

  "client": {
    "name": "string",
    "type": "string | null",
    "jurisdiction": "string | null",
    "address": "string | null",
    "signatory": "string | null",
    "signatory_title": "string | null",
    "signatory_date": "string | null — ISO 8601 date"
  },

  "scope_of_services": "string — faithful description of services from the document",

  "term": {
    "duration_months": "number | null",
    "start_date": "string | null — ISO 8601",
    "end_date": "string | null — ISO 8601",
    "auto_renew": "boolean — false if not mentioned",
    "renewal_term_months": "number | null",
    "non_renewal_notice_days": "number | null"
  },

  "compensation": {
    "fee_type": "string | null — hourly, monthly retainer, fixed project, milestone-based, retainer, percentage, per-unit, revenue sharing, in-kind",
    "monthly_fee": "number | null",
    "hourly_rate": "number | null",
    "total_value": "number | null",
    "setup_fee": "number | null",
    "currency": "string — default USD",
    "payment_terms_days": "number | null",
    "payment_terms_note": "string | null — e.g. 'business days', 'in arrears', 'monthly in advance'",
    "late_payment_interest_rate_percent": "number | null",
    "max_monthly_hours": "number | null",
    "monthly_expense_cap": "number | null",
    "per_unit_rate": "number | null",
    "monthly_volume_cap": "number | null",
    "payment_schedule": "array | null — [{trigger: string, percentage: number | null, amount: number}]",
    "payment_method": "string | null — e.g. 'wire transfer', 'ACH'"
  },

  "deliverables_or_milestones": [
    {"description": "string", "due_within_days": "number | null", "due_date": "string | null — ISO 8601"}
  ],

  "intellectual_property": {
    "ownership": "string | null — e.g. 'Service Provider', 'Client', 'Client (upon payment)', 'Joint'",
    "license_granted": "boolean | null",
    "license_type": "string | null — e.g. 'non-exclusive, non-transferable', 'perpetual, irrevocable'",
    "license_scope": "string | null",
    "exceptions": "string | null"
  },

  "confidentiality": {
    "duration_years_after_termination": "number | null"
  },

  "termination": {
    "notice_period_days": "number | null — general termination notice",
    "for_cause_cure_period_days": "number | null",
    "for_convenience_notice_days": "number | null",
    "for_convenience_notice_days_client": "number | null",
    "for_convenience_notice_days_provider": "number | null"
  },

  "liability": {
    "cap_type": "string | null — description of how cap is calculated",
    "cap_amount": "number | null",
    "consequential_damages_excluded": "boolean | null"
  },

  "insurance": {
    "general_liability": "number | null",
    "professional_liability_per_occurrence": "number | null",
    "professional_liability_aggregate": "number | null"
  },

  "warranty": {
    "defect_free_period_months": "number | null",
    "ip_non_infringement": "boolean | null"
  },

  "dispute_resolution": {
    "method": "string | null — litigation, arbitration, mediation",
    "location": "string | null",
    "arbitration_body": "string | null — e.g. 'AAA', 'JAMS'"
  },

  "non_solicitation": {
    "duration_months_after_termination": "number | null"
  },

  "governing_law": {
    "jurisdiction": "string | null",
    "country": "string | null — default USA"
  }
}
```

#### Schema Extensions

Domain-specific fields may be added to the JSON as needed. These are not part of the base schema but are useful for capturing unique contract structures:

| Field | Used In | Description |
|-------|---------|-------------|
| `compensation.success_bonus` | sa_007 | Performance bonus amount |
| `compensation.ad_spend_budget_monthly` | sa_004 | Managed advertising spend |
| `compensation.per_employee_rate` | sa_008 | Per-employee pricing |
| `compensation.tiered_pricing` | sa_010 | Array of pricing tiers |
| `compensation.rush_surcharge_percent` | sa_010 | Rush order surcharge |
| `compensation.minimum_monthly_engagement` | sa_010 | Minimum monthly billing |
| `compensation.retainage_percent` | sa_015 | Retainage held until completion |
| `compensation.minimum_annual_guarantee` | sa_018 | Minimum payment to client |
| `termination.kpi_based_termination` | sa_004 | Boolean for KPI-triggered termination |
| `termination.kpi_termination_cure_days` | sa_004 | Cure period before KPI termination |
| `termination.per_phase_termination_allowed` | sa_011 | Per-phase/matter termination right |
| `termination.bankruptcy_termination` | sa_020 | Immediate termination on bankruptcy |
| `liability.per_matter_liability_cap` | sa_009 | Per-matter cap (litigation support) |
| `liability.aggregate_liability_cap` | sa_009 | Aggregate cap across all matters |
| `insurance.cyber_liability` | sa_006 | Cyber liability coverage |
| `insurance.liquor_liability` | sa_018 | Liquor liability coverage |
| `insurance.pollution_liability` | sa_011 | Pollution/environmental liability |
| `insurance.fidelity_bond` | sa_017 | Fidelity bond amount |
| `insurance.employment_practices_liability` | sa_008 | EPL coverage |
| `secondary_client` | sa_020 | Third-party in multi-party agreements |

**Rule**: Only add schema extensions when the contract structure genuinely requires it. Do not add fields for values that fit into the base schema.

#### Extraction Rules

1. **Null for missing**: Use `null` for any field not mentioned or not derivable from the document. Do **not** guess or infer. If an agreement number is redacted, use `null` or the string `"REDACTED"`.
2. **Extract, don't summarize**: For `scope_of_services`, use a concise but faithful description of what the document says. Do not paraphrase or add context not present in the document.
3. **Normalize dates**: Always convert to `YYYY-MM-DD` ISO format. If only a relative timeframe is given (e.g., "within 30 days"), use `due_within_days` instead. If the effective date is ambiguous (e.g., "upon the later of two conditions"), look for the actual execution date buried in signature blocks or footnotes.
4. **Normalize amounts**: Extract as numbers (no currency symbols, no commas). For ranges, extract the upper bound. For percentage-based fees, extract the percentage in the relevant field and describe the basis in `cap_type` or `payment_terms_note`.
5. **Boolean defaults**:
   - `auto_renew`: `false` if not mentioned.
   - `consequential_damages_excluded`: `true` only if the document explicitly excludes them; otherwise `null`.
   - `license_granted`: `null` if no license is mentioned, `true`/`false` based on the document.
6. **Ambiguity**: If a value is ambiguous (e.g., a payment term could be calendar days or business days but the document doesn't specify), extract what is stated and note the ambiguity in the relevant `_note` field if one exists, or leave it as-is.
7. **Party roles**: Always map the party providing services to `service_provider` and the party receiving services to `client`, regardless of the labels used in the document (Contractor/Client, Vendor/Customer, Consultant/Company, Agency/Brand, Lead Research Org/Industry Sponsor, etc.).
8. **Multi-party agreements**: For agreements with more than two parties, map the primary service provider to `service_provider`, the primary client/sponsor to `client`, and use `secondary_client` for additional parties.
9. **In-kind contributions**: When compensation includes non-cash contributions (equipment, personnel, facilities), note them in `payment_terms_note` and do not convert to a dollar amount unless the document explicitly values them.
10. **Contingent fees**: For fees that depend on future conditions (e.g., "Phase II fee contingent on Phase I findings"), extract the stated amount and note the contingency in `payment_terms_note`.

### Step 3: Validate

- [ ] Verify the JSON file is valid JSON (no trailing commas, proper quoting, valid types).
- [ ] Cross-check that every value in the JSON can be traced back to specific text in the input document.
- [ ] Verify the agreement number matches between input and expected files.
- [ ] Ensure the key `deliverables_or_milestones` is used consistently (not `milestones`, `deliverables`, or `deliverables_or_milestones` interchangeably).
- [ ] Verify all dates are in `YYYY-MM-DD` ISO format.
- [ ] Verify all monetary amounts are plain numbers (no `$`, no commas).
- [ ] Check that `null` is used for genuinely missing fields — not empty strings, not `"N/A"`, not `"not specified"`.
- [ ] Confirm the input file is 250+ lines with adequate dilution.

---

## Generating Data at Scale

When generating large batches of examples, use LLM-based agents (3 concurrent max) with a detailed prompt that includes:

1. **The full schema** from this guide (both base + relevant extensions).
2. **The dilution requirements** — specify 250–800 lines, 75–85% noise, specific dilution techniques.
3. **Edge case assignments** — deliberately distribute edge cases across the batch so no two consecutive examples share the same challenge pattern.
4. **Industry variety** — rotate through different industries to ensure domain diversity.
5. **Consistency instructions** — explicitly state schema key names to avoid drift (e.g., always `deliverables_or_milestones`, never `milestones` or `deliverables`).
6. **A "hard mode" example** — reference sa_020 as the difficulty ceiling for the most challenging examples in the set.

**Quality gate**: After generation, run a validation pass that:
- Checks line counts (250–800 lines)
- Validates all JSON files parse correctly
- Verifies agreement numbers exist in both input and expected
- Flags any expected JSON that is "too clean" (all fields populated, no nulls) — this likely means insufficient edge cases

---

## Expanding to New Categories

To create examples for a different document category:

1. Create a new subdirectory under `examples/` (e.g., `examples/invoices/`)
2. Define a schema in a `SCHEMA.md` file for that category
3. Create `inputs/` and `expected/` subdirectories
4. Follow the same naming convention (`cat_NNN.txt` / `cat_NNN.json`)
5. Adapt this guide's dilution requirements to the new document type
6. Define category-specific edge cases

---

## Current Coverage

| Category | Directory | Count | Status |
|----------|-----------|-------|--------|
| Service Agreements | `service_agreements/` | 20 | Complete |

## Document Index

| # | File | Industry | Lines | Key Challenges |
|---|------|----------|-------|----------------|
| 001 | sa_001 | Cloud infrastructure | 331 | Standard baseline dilution |
| 002 | sa_002 | Healthcare consulting | 415 | Auto-renewal, hourly rates, dual-party termination |
| 003 | sa_003 | Software development (fintech) | 397 | Fixed project, phased payments, international governing law |
| 004 | sa_004 | Digital marketing | 661 | KPI-based termination, retainer + ad spend |
| 005 | sa_005 | Logistics/warehouse | 663 | Per-unit pricing, volume caps, SLA-based termination |
| 006 | sa_006 | Cybersecurity (banking) | 683 | Setup fee + monthly, 10-year confidentiality, federal courts |
| 007 | sa_007 | Construction project mgmt | 730 | Fixed fee with success bonus, LEED certification |
| 008 | sa_008 | HR outsourcing | 746 | Per-employee pricing, implementation fee, FMLA/ADA compliance |
| 009 | sa_009 | Legal research/litigation | 733 | Tiered hourly rates, per-matter termination, blended rate cap |
| 010 | sa_010 | Translation/localization | 736 | Tiered per-word pricing by language, rush surcharge, TM ownership |
| 011 | sa_011 | Environmental consulting | 787 | Contingent Phase II fee, field day rates, per-phase termination |
| 012 | sa_012 | Financial audit | 762 | PCAOB/SOX references, successor auditor consent, engagement letter |
| 013 | sa_013 | Data analytics/BI | 649 | Overtime threshold, one-time infrastructure fee, data governance |
| 014 | sa_014 | Event management (pharma) | 690 | One-event agreement, production budget with line-item approval, FDA compliance |
| 015 | sa_015 | Architectural/engineering | 674 | Percentage-based fees, retainage, copyright retention, nonprofit client |
| 016 | sa_016 | IT staff augmentation | 701 | Blended + specialist rates, overtime, minimum billable hours, anti-poaching |
| 017 | sa_017 | Property management | 703 | Revenue percentage fees, leasing commissions, REIT client, per-property removal |
| 018 | sa_018 | Food/beverage (venue) | 782 | Revenue sharing model, minimum guarantee, liquor liability, joint powers authority |
| 019 | sa_019 | Telecom infrastructure | 748 | Per-premises fee, federal subsidy, 60-month term, FCC compliance |
| 020 | sa_020 | R&D consortium | 719 | REDACTED agreement number, three parties, in-kind contributions, joint IP, no dispute resolution, no insurance |

**Total**: 20 documents, 13,381 lines of input text, 20 expected extractions.
