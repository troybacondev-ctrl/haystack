"""Runnable sponsor matching pipeline using only synthetic data.

This example demonstrates a practical handoff point for Gemini structured output:
retrieval and prompt construction are real Haystack components, while the final
matching step is deterministic so this file runs without credentials or network
access. A Gemini generator can later consume ``prompt`` and return the same
structured match schema.

Attribution: downstream fork contribution by Troy Bacon.
"""

from __future__ import annotations

from typing import Any

from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


PROMPT_TEMPLATE = """You are matching an applicant to sponsorship opportunities.
Use only the retrieved opportunities below. Eligibility has been checked by a
separate deterministic step; return JSON with a `matches` array containing
`sponsor_name`, `fit_score`, and `explanation`.

Applicant:
- Organization: {{ applicant.organization }}
- Region: {{ applicant.region }}
- Industry: {{ applicant.industry }}
- Annual revenue: ${{ applicant.annual_revenue }}
- Team size: {{ applicant.team_size }}

Retrieved opportunities:
{% for document in documents %}
- {{ document.meta.sponsor_name }}: {{ document.content }}
{% endfor %}

Request: {{ query }}
"""


@component
class DeterministicSponsorMatcher:
    """Filter eligible sponsors and rank them without making an LLM call."""

    @component.output_types(matches=list[dict[str, Any]], prompt=str)
    def run(
        self,
        documents: list[Document],
        applicant: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        """Return eligible sponsors with deterministic scores and explanations.

        ``prompt`` is intentionally accepted but not interpreted. In a production
        pipeline, this component is the integration point for a Gemini structured
        output generator configured with a JSON schema for the match records.
        """
        matches: list[dict[str, Any]] = []

        for document in documents:
            metadata = document.meta
            reasons: list[str] = []
            region = applicant["region"]
            industry = applicant["industry"]
            revenue = applicant["annual_revenue"]
            team_size = applicant["team_size"]

            if region not in metadata["regions"]:
                continue
            reasons.append(f"serves {region}")
            if industry not in metadata["industries"]:
                continue
            reasons.append(f"supports {industry}")
            if revenue < metadata["min_annual_revenue"]:
                continue
            if revenue > metadata["max_annual_revenue"]:
                continue
            reasons.append("revenue is within the sponsor range")
            if team_size < metadata["min_team_size"]:
                continue
            reasons.append("meets the minimum team size")

            retrieval_score = document.score or 0.0
            fit_score = round(70 + min(retrieval_score, 10.0) * 2 + len(reasons) * 5, 2)
            matches.append(
                {
                    "sponsor_name": metadata["sponsor_name"],
                    "fit_score": fit_score,
                    "retrieval_score": round(retrieval_score, 3),
                    "explanation": "; ".join(reasons) + ".",
                }
            )

        matches.sort(key=lambda match: (-match["fit_score"], match["sponsor_name"]))
        return {"matches": matches, "prompt": prompt}


def build_pipeline() -> Pipeline:
    """Build the retrieval, prompt-building, and deterministic matching graph."""
    documents = [
        Document(
            content="Climate technology sponsorship for measurable emissions reduction projects.",
            meta={
                "sponsor_name": "Green Horizon Fund",
                "regions": ["North America", "Europe"],
                "industries": ["climate", "energy"],
                "min_annual_revenue": 100_000,
                "max_annual_revenue": 20_000_000,
                "min_team_size": 3,
            },
        ),
        Document(
            content="Community health sponsorship for scalable healthcare access programs.",
            meta={
                "sponsor_name": "WellSpring Partners",
                "regions": ["North America"],
                "industries": ["healthcare"],
                "min_annual_revenue": 250_000,
                "max_annual_revenue": 50_000_000,
                "min_team_size": 5,
            },
        ),
        Document(
            content="Inclusive technology sponsorship for open-source software and digital inclusion.",
            meta={
                "sponsor_name": "Open Bridge Initiative",
                "regions": ["North America", "Europe"],
                "industries": ["technology", "education"],
                "min_annual_revenue": 50_000,
                "max_annual_revenue": 10_000_000,
                "min_team_size": 2,
            },
        ),
        Document(
            content="Local arts sponsorship for cultural events and creative education.",
            meta={
                "sponsor_name": "Civic Arts Circle",
                "regions": ["North America"],
                "industries": ["arts", "education"],
                "min_annual_revenue": 25_000,
                "max_annual_revenue": 5_000_000,
                "min_team_size": 1,
            },
        ),
    ]
    store = InMemoryDocumentStore()
    store.write_documents(documents)

    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=4))
    pipeline.add_component("prompt_builder", PromptBuilder(template=PROMPT_TEMPLATE))
    pipeline.add_component("matcher", DeterministicSponsorMatcher())
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("retriever.documents", "matcher.documents")
    pipeline.connect("prompt_builder.prompt", "matcher.prompt")
    return pipeline


def main() -> None:
    """Run a synthetic sponsor matching request and print its results."""
    applicant = {
        "organization": "Northstar Retrofit Co-op",
        "region": "North America",
        "industry": "climate",
        "annual_revenue": 2_000_000,
        "team_size": 8,
    }
    query = "climate technology sponsorship for emissions reduction"
    result = build_pipeline().run(
        {
            "retriever": {"query": query},
            "prompt_builder": {"applicant": applicant, "query": query},
            "matcher": {"applicant": applicant},
        }
    )

    print("Prompt sent to a future Gemini structured-output generator:\n")
    print(result["matcher"]["prompt"])
    print("Deterministic matches (no API key required):")
    for match in result["matcher"]["matches"]:
        print(f"- {match['sponsor_name']} ({match['fit_score']}): {match['explanation']}")


if __name__ == "__main__":
    main()
