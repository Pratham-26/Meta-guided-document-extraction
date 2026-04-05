# Example Data Creation Guide

## Purpose

This directory contains sample **service agreement** documents for the meta-learning document extraction system. These examples serve as review/reference material — they are **not** consumed by the system directly.

## Directory Structure

```
examples/
└── service_agreements/
    ├── inputs/          # Raw text files (sa_001.txt, sa_002.txt, ...)
    └── DATA_GUIDE.md    # This file
```

## File Naming Convention

- **Inputs**: `sa_NNN.txt` where NNN is a zero-padded 3-digit sequential number (001–999)

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

### Step 2: Validate

- [ ] Verify the agreement number is present in the input document.
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
- Verifies agreement numbers exist in inputs

---

## Expanding to New Categories

To create examples for a different document category:

1. Create a new subdirectory under `examples/` (e.g., `examples/invoices/`)
2. Define a schema in a `SCHEMA.md` file for that category
3. Create `inputs/` subdirectory
4. Follow the same naming convention (`cat_NNN.txt`)
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

**Total**: 20 documents, 13,381 lines of input text.
