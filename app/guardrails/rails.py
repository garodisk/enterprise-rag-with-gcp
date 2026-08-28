import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses llama-3.1-8b-instant for fast intent classification at the gate —
    the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_MODEL}).")
    
    


# def guard(message: str) -> tuple[bool, str | None]:
#     """
#     Run a user message through the NeMo rails gate.

#     Returns:
#         (True,  rail_response) — a rail fired; return this response immediately,
#                                 skip the RAG pipeline entirely.
#         (False, None)          — message is clean; proceed to LangGraph.
#     """
#     if _rails is None:
#         logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
#         return False, None

#     with logfire.span("🛡️ Guardrails Check"):
#         result = _rails.generate(messages=[{"role": "user", "content": message}])

#         # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
#         content = result.get("content", "") if isinstance(result, dict) else str(result)

#         fired = any(indicator in content for indicator in RAIL_INDICATORS)

#         if fired:
#             logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
#             return True, content

#         logfire.info("✅ Guardrails passed.")
#         return False, None


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True, rail_response)  -> a rail fired; return immediately
        (False, None)          -> no rail fired; continue to LangGraph
    """

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": message
                }
            ]
        )

        # Extract assistant text from NeMo result
        if isinstance(result, dict):
            content = result.get("content", "")
        else:
            content = str(result)

        # -----------------------------
        # TEMP DEBUGGING
        # -----------------------------
        print("\n========== NEMO DEBUG ==========")
        print("USER:")
        print(message)

        print("\nRAW RESULT:")
        print(result)

        print("\nCONTENT:")
        print(content)

        print("\nINDICATOR MATCHES:")
        for indicator in RAIL_INDICATORS:
            matched = indicator in content
            print(f"{repr(indicator)} -> {matched}")

        # Current rail-detection logic
        fired = any(
            indicator in content
            for indicator in RAIL_INDICATORS
        )

        print("\nFINAL FIRED:")
        print(fired)

        print("================================\n")

        # Also capture useful info in Logfire
        logfire.info(
            "🛡️ NeMo result evaluated",
            user_message=message[:100],
            rail_fired=fired,
            nemo_content=content[:500],
        )

        if fired:
            logfire.info(
                f"🛡️ Guardrails fired | query='{message[:80]}'"
            )

            return True, content

        logfire.info("✅ Guardrails passed.")

        return False, None
