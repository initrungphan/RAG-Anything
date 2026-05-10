#!/usr/bin/env python
"""
Query-only runner for an existing LightRAG / RAGAnything working directory.

Use this after you have already indexed documents (e.g. via
``raganything_example.py``). It does **not** parse or ingest files.

Examples
--------

From the repository root, with ``.env`` containing ``LLM_BINDING_API_KEY`` and the
same ``EMBEDDING_MODEL`` / ``EMBEDDING_DIM`` used when building the index::

    python examples/query_storage.py -w ./rag_storage_history_only

Multiple queries and substring checks (one ``--expect`` per ``-q``; order must match)::

    python examples/query_storage.py -w ./rag_storage_history_only \\
      -q "For 08-APR-26, quote the B_FORE forecast row from the document." \\
      -q "What REC_TYPE values appear in the data?" \\
      --expect "B_FORE" \\
      --expect "B_FORE"

Queries from a file (one query per line; ``#`` starts a comment line)::

    python examples/query_storage.py -w ./rag_storage_history_only --queries-file my_queries.txt

Exit codes
----------

- ``0``: all queries ran; optional substring checks passed.
- ``1``: bad arguments, LightRAG init failure, or a substring check failed.

Caveats
-------

- The ``working_dir`` must already contain a completed index; mismatched
  embedding model or dimension will give poor or broken retrieval.
- ``_ensure_lightrag_initialized`` still verifies parser installation (e.g.
  MinerU); keep the same ``--parser`` you used when indexing if relevant.
- Substring checks are best-effort: the LLM may paraphrase; prefer stable
  tokens present in the source (e.g. ``B_FORE``, ``08-APR-26``).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from functools import partial
from pathlib import Path

# Repository root on sys.path (same pattern as raganything_example.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger
from raganything import RAGAnything, RAGAnythingConfig

DEFAULT_QUERIES = [
    "What is the main content of the document?",
    (
        "For date 2026-04-28, return the full data row "
        "from the document with all pipe-separated fields. Cite the source file name."
    ),
    "What are the key topics discussed in the forecast and history rows?",
]


def _load_queries_from_file(path: Path) -> list[str]:
    lines: list[str] = []
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _build_rag(
    *,
    working_dir: str,
    parser: str,
    api_key: str,
    base_url: str | None,
) -> RAGAnything:
    config = RAGAnythingConfig(
        working_dir=working_dir,
        parser=parser,
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    vision_model = os.getenv("VISION_MODEL", "gpt-4o")

    def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ):
        history_messages = history_messages or []
        return openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        image_data=None,
        messages=None,
        **kwargs,
    ):
        history_messages = history_messages or []
        if messages:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        if image_data:
            return openai_complete_if_cache(
                vision_model,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=base_url,
        ),
    )

    return RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )


async def _run_queries(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        logger.error(
            "API key required: set LLM_BINDING_API_KEY or pass --api-key",
        )
        return 1

    if args.queries_file and args.queries:
        logger.error("Use either --queries-file or -q/--query, not both.")
        return 1

    if args.queries_file:
        qpath = Path(args.queries_file)
        if not qpath.is_file():
            logger.error(f"Queries file not found: {qpath}")
            return 1
        queries = _load_queries_from_file(qpath)
        if not queries:
            logger.error(f"No queries in file after skipping comments/empties: {qpath}")
            return 1
    elif args.queries:
        queries = args.queries
    else:
        queries = list(DEFAULT_QUERIES)

    expects = args.expects or []
    if expects and len(expects) != len(queries):
        logger.error(
            f"Number of --expect ({len(expects)}) must match number of queries ({len(queries)}).",
        )
        return 1

    rag = _build_rag(
        working_dir=args.working_dir,
        parser=args.parser,
        api_key=api_key,
        base_url=args.base_url,
    )

    try:
        init = await rag._ensure_lightrag_initialized()
        if not init or not init.get("success"):
            err = (init or {}).get("error", "unknown error")
            logger.error(f"LightRAG initialization failed: {err}")
            return 1

        all_passed = True
        for i, query in enumerate(queries):
            label = f"Query {i + 1}/{len(queries)}"
            logger.info(f"\n[{label}]: {query}")
            answer = await rag.aquery(
                query,
                mode=args.mode,
                vlm_enhanced=args.vlm_enhanced,
            )
            print(answer)
            print("-" * 60)

            if expects:
                exp = expects[i]
                if exp and exp not in answer:
                    logger.error(
                        f"{label} substring check failed: expected {exp!r} in answer.",
                    )
                    all_passed = False

        return 0 if all_passed else 1
    finally:
        rag.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run text queries against an existing RAGAnything / LightRAG working directory.",
    )
    parser.add_argument(
        "--working-dir",
        "-w",
        default="./rag_storage",
        help="LightRAG working directory (must already be indexed)",
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Parser name (installation is still checked on init)",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        help="LightRAG query mode (e.g. hybrid, local, global, naive, mix)",
    )
    parser.add_argument(
        "--vlm-enhanced",
        action="store_true",
        help="Use VLM-enhanced query path (default: plain text aquery)",
    )
    parser.add_argument(
        "-q",
        "--query",
        action="append",
        dest="queries",
        default=None,
        help="Query string (repeat -q for multiple). Default: built-in sample queries.",
    )
    parser.add_argument(
        "--queries-file",
        metavar="PATH",
        help="File with one query per line (# comments allowed)",
    )
    parser.add_argument(
        "--expect",
        action="append",
        dest="expects",
        default=None,
        help=(
            "Substring that must appear in the answer for the query at the same "
            "position (repeat --expect; count must match -q / file lines)"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_BINDING_API_KEY"),
        help="LLM API key (default: LLM_BINDING_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Optional API base URL (default: LLM_BINDING_HOST)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(_run_queries(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
