import dspy
import json


class ExtractDocument(dspy.Signature):
    """Extract structured data from retrieved document context. Follow the instructions
    precisely and output valid JSON matching the provided schema."""

    context: str = dspy.InputField(
        desc="Retrieved document pages or chunks relevant to extraction"
    )
    extraction_schema: str = dspy.InputField(
        desc="JSON schema defining expected output fields and types"
    )
    extraction_instructions: str = dspy.InputField(
        desc="Extraction instructions for this document category"
    )
    few_shot_examples: str = dspy.InputField(
        desc="Example extractions from Gold Standards, or empty string"
    )
    extraction: str = dspy.OutputField(
        desc="Extracted data as a JSON object matching the schema exactly"
    )


class ExtractorAgent:
    def __init__(self, lm: dspy.LM | None = None):
        if lm:
            dspy.configure(lm=lm)
        self.extract = dspy.Predict(ExtractDocument)

    def run(
        self,
        context: str,
        schema: dict,
        instructions: str,
        few_shot_examples: list[dict] | None = None,
    ) -> dict:
        examples_str = ""
        if few_shot_examples:
            examples_str = json.dumps(few_shot_examples, indent=2)

        result = self.extract(
            context=context,
            extraction_schema=json.dumps(schema, indent=2),
            extraction_instructions=instructions,
            few_shot_examples=examples_str,
        )

        try:
            extraction = json.loads(result.extraction)
        except json.JSONDecodeError:
            extraction = {"raw": result.extraction, "_parse_error": True}

        return extraction
