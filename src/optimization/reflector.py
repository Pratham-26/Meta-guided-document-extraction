import dspy
import json


class ReflectOnFailure(dspy.Signature):
    """Analyze why an extraction failed compared to the Gold Standard.
    Diagnose root causes at the instruction level — what about the current instructions
    led to the incorrect or missing extraction."""

    extraction: str = dspy.InputField(desc="The failed extraction result (JSON)")
    gold_standard: str = dspy.InputField(
        desc="The correct Gold Standard extraction (JSON)"
    )
    extraction_schema: str = dspy.InputField(desc="JSON schema for the extraction")
    current_instructions: str = dspy.InputField(
        desc="Current extraction instructions being used"
    )
    field_feedback: str = dspy.InputField(
        desc="Per-field feedback from the Judge Agent"
    )
    diagnosis: str = dspy.OutputField(
        desc="Root cause analysis: why did each field fail? What instruction gaps caused this?"
    )
    suggested_fixes: str = dspy.OutputField(
        desc="Specific, actionable instruction changes that would fix these failures"
    )


class MutatePrompt(dspy.Signature):
    """Given a diagnosis of extraction failures and suggested fixes, produce revised
    extraction instructions. Make targeted, minimal changes — don't rewrite everything."""

    current_instructions: str = dspy.InputField(desc="Current extraction instructions")
    diagnosis: str = dspy.InputField(desc="Root cause analysis of failures")
    suggested_fixes: str = dspy.InputField(desc="Suggested instruction changes")
    revised_instructions: str = dspy.OutputField(
        desc="Improved extraction instructions"
    )
    mutation_rationale: str = dspy.OutputField(
        desc="Brief explanation of what changed and why"
    )


class Reflector:
    def __init__(self, lm: dspy.LM | None = None):
        if lm:
            dspy.configure(lm=lm)
        self.reflect = dspy.Predict(ReflectOnFailure)

    def analyze(
        self,
        extraction: dict,
        gold_standard: dict,
        schema: dict,
        current_instructions: str,
        field_feedback: str,
    ) -> dict:
        result = self.reflect(
            extraction=json.dumps(extraction, indent=2),
            gold_standard=json.dumps(gold_standard, indent=2),
            extraction_schema=json.dumps(schema, indent=2),
            current_instructions=current_instructions,
            field_feedback=field_feedback,
        )
        return {
            "diagnosis": result.diagnosis,
            "suggested_fixes": result.suggested_fixes,
        }


class PromptMutator:
    def __init__(self, lm: dspy.LM | None = None):
        if lm:
            dspy.configure(lm=lm)
        self._mutate_prompt = dspy.Predict(MutatePrompt)

    def mutate(
        self,
        current_instructions: str,
        diagnosis: str,
        suggested_fixes: str,
    ) -> dict:
        result = self._mutate_prompt(
            current_instructions=current_instructions,
            diagnosis=diagnosis,
            suggested_fixes=suggested_fixes,
        )
        return {
            "revised_instructions": result.revised_instructions,
            "mutation_rationale": result.mutation_rationale,
        }
