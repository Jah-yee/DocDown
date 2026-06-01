"""CLI entry point for DocDown."""

from pathlib import Path
from time import perf_counter

import click

from docdown.config import ConfigError, load_config
from docdown.stages.chunk_validation import ChunkResult, validate_chunk
from docdown.stages.cleanup import CleanupError, cleanup_markdown_file
from docdown.stages.convert import PandocError, convert_to_markdown, ensure_pandoc_available
from docdown.stages.extract import ExtractorUsed, orchestrate_extraction
from docdown.stages.final_validation import FinalValidationError, validate_final_output
from docdown.stages.merge import MergeError, merge_chunks
from docdown.stages.run_summary import (
    RunSummaryContext,
    RunSummaryError,
    append_run_summary,
    generate_run_summary,
)
from docdown.stages.split import PdfSplitError, PdfValidationError, split_pdf, validate_pdf
from docdown.stages.toc import TocError, generate_toc, log_heading_diagnostics
from docdown.utils.logging import configure_logging
from docdown.workdir import WorkDir, WorkDirError


@click.command()
@click.argument("input_pdf", required=False, type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option("--config", "-c", type=click.Path(), default=None, help="Path to docdown.yaml")
@click.option(
    "--workdir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Working directory for artifacts",
)
@click.option("--chunk-size", type=int, default=None, help="Pages per chunk")
@click.option("--parallel-workers", type=int, default=None, help="Max chunk workers")
@click.option("--extractor", type=click.Choice(["grobid", "pdfminer"]), default=None, help="Primary extractor")
@click.option(
    "--fallback-extractor",
    type=click.Choice(["grobid", "pdfminer"]),
    default=None,
    help="Fallback extractor",
)
@click.option("--grobid-url", type=str, default=None, help="GROBID service URL")
@click.option("--table-extraction/--no-table-extraction", default=None, help="Enable table extraction")
@click.option("--llm-cleanup/--no-llm-cleanup", default=None, help="Enable LLM cleanup pipeline")
@click.option("--llm-model", type=str, default=None, help="LLM model identifier")
@click.option(
    "--heuristic-numbered-headings/--no-heuristic-numbered-headings",
    default=None,
    help="Promote numbered section-title lines to headings during cleanup",
)
@click.option(
    "--heuristic-titlecase-headings/--no-heuristic-titlecase-headings",
    default=None,
    help="Promote probable title-case section lines to headings during cleanup",
)
@click.option(
    "--heuristic-allcaps-headings/--no-heuristic-allcaps-headings",
    default=None,
    help="Promote probable ALL CAPS section lines to headings during cleanup",
)
@click.option("--toc-depth", type=click.IntRange(1, 6), default=None, help="TOC maximum heading depth (1-6)")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "WARN"], case_sensitive=False),
    default=None,
    help="Log verbosity level",
)
@click.option("--min-output-ratio", type=float, default=None, help="Validation minimum output ratio")
@click.option("--max-empty-chunks", type=int, default=None, help="Validation max empty chunks")
def main(
    input_pdf,
    config,
    workdir,
    chunk_size,
    parallel_workers,
    extractor,
    fallback_extractor,
    grobid_url,
    table_extraction,
    llm_cleanup,
    llm_model,
    heuristic_numbered_headings,
    heuristic_titlecase_headings,
    heuristic_allcaps_headings,
    toc_depth,
    log_level,
    min_output_ratio,
    max_empty_chunks,
):
    """Convert a PDF to Markdown."""

    run_start = perf_counter()

    cli_overrides = {
        "input": input_pdf,
        "workdir": workdir,
        "chunk_size": chunk_size,
        "parallel_workers": parallel_workers,
        "extractor": extractor,
        "fallback_extractor": fallback_extractor,
        "grobid_url": grobid_url,
        "table_extraction": table_extraction,
        "llm_cleanup": llm_cleanup,
        "llm_model": llm_model,
        "heuristic_numbered_headings": heuristic_numbered_headings,
        "heuristic_titlecase_headings": heuristic_titlecase_headings,
        "heuristic_allcaps_headings": heuristic_allcaps_headings,
        "toc_depth": toc_depth,
        "log_level": log_level,
        "validation": {
            "min_output_ratio": min_output_ratio,
            "max_empty_chunks": max_empty_chunks,
        },
    }

    try:
        cfg = load_config(config_path=_resolve_config_path(config), cli_overrides=cli_overrides)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    if cfg.input is None:
        raise click.ClickException("No input PDF provided. Pass INPUT_PDF or set 'input' in docdown.yaml.")

    try:
        work_dir = WorkDir(cfg.workdir)
        work_dir.ensure_structure()
        staged_input = work_dir.stage_input(cfg.input)
    except WorkDirError as exc:
        raise click.ClickException(str(exc)) from exc

    logger = configure_logging(cfg.workdir, cfg.log_level)
    version_text = f"DocDown v{_version()}"
    input_text = f"Input:   {cfg.input}"
    workdir_text = f"Workdir: {cfg.workdir}"

    # Keep CLI summaries on stdout while also recording them in logs.
    click.echo(version_text)
    click.echo(input_text)
    click.echo(workdir_text)

    logger.info(version_text)
    logger.info(input_text)
    logger.info(workdir_text)
    logger.debug("Staged input at %s", staged_input)

    try:
        validation = validate_pdf(staged_input, logger=logger)
    except PdfValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    logger.info("PDF ready for splitting: pages=%s size_bytes=%s", validation.page_count, validation.file_size_bytes)

    try:
        split_result = split_pdf(
            staged_input,
            work_dir.chunks_dir,
            cfg.chunk_size,
            validation.page_count,
            logger=logger,
        )
    except PdfSplitError as exc:
        raise click.ClickException(str(exc)) from exc

    logger.info("Split complete: %s chunks", split_result.chunk_count)

    extraction_results = orchestrate_extraction(
        split_result.chunk_paths,
        work_dir.extracted_dir,
        extractor=cfg.extractor,
        fallback_extractor=cfg.fallback_extractor,
        grobid_url=cfg.grobid_url,
        logger=logger,
    )

    extraction_failure_results = [
        ChunkResult(
            chunk_number=result.chunk_number,
            success=False,
            markdown_path=None,
            error=getattr(result, "error", None),
            validation=None,
        )
        for result in extraction_results
        if not result.success
    ]

    successful_extractions = [
        result for result in extraction_results if result.success and result.output_path is not None
    ]
    if not successful_extractions:
        raise click.ClickException("Extraction failed for all chunks; no conversion input available.")

    try:
        ensure_pandoc_available(logger=logger)
    except PandocError as exc:
        raise click.ClickException(str(exc)) from exc

    converted_chunks = 0
    converted_before_validation = 0
    chunk_results: list[ChunkResult] = []
    for result in successful_extractions:
        markdown_path = work_dir.markdown(result.chunk_number)
        try:
            convert_to_markdown(
                result.output_path,
                markdown_path,
                logger=logger,
                chunk_number=result.chunk_number,
            )
            cleanup_markdown_file(
                markdown_path,
                logger=logger,
                chunk_number=result.chunk_number,
                heuristic_numbered_headings=cfg.heuristic_numbered_headings,
                heuristic_titlecase_headings=cfg.heuristic_titlecase_headings,
                heuristic_allcaps_headings=cfg.heuristic_allcaps_headings,
            )
        except UnicodeDecodeError:
            logger.error("Markdown conversion/cleanup failed for chunk-%04d: Invalid UTF-8 encoding", result.chunk_number)
            chunk_results.append(
                ChunkResult(
                    chunk_number=result.chunk_number,
                    success=False,
                    markdown_path=markdown_path,
                    error="Invalid UTF-8 encoding",
                    validation=None,
                )
            )
            continue
        except (PandocError, CleanupError, OSError) as exc:
            logger.error("Markdown conversion/cleanup failed for chunk-%04d: %s", result.chunk_number, exc)
            chunk_results.append(
                ChunkResult(
                    chunk_number=result.chunk_number,
                    success=False,
                    markdown_path=None,
                    error=str(exc),
                    validation=None,
                )
            )
            continue

        converted_before_validation += 1

        extractor_used = getattr(result, "extractor", None)
        if isinstance(extractor_used, ExtractorUsed):
            extractor_name = extractor_used.value
        else:
            extractor_name = str(extractor_used) if extractor_used is not None else None

        if not (1 <= result.chunk_number <= len(split_result.chunk_paths)):
            raise click.ClickException(
                f"Chunk number {result.chunk_number} out of range (1..{len(split_result.chunk_paths)})"
            )

        chunk_validation = validate_chunk(
            markdown_path,
            split_result.chunk_paths[result.chunk_number - 1],
            min_output_ratio=cfg.validation.min_output_ratio,
            expect_headings=extractor_name != ExtractorUsed.PDFMINER.value,
            logger=logger,
            chunk_number=result.chunk_number,
        )
        if not chunk_validation.valid:
            recoverable_validation_errors = {"Empty output", "Invalid UTF-8 encoding"}
            non_recoverable_errors = [
                issue for issue in chunk_validation.errors if issue not in recoverable_validation_errors
            ]
            if non_recoverable_errors:
                raise click.ClickException(
                    "Chunk validation failed for "
                    f"chunk-{result.chunk_number:04d}: "
                    f"{' ; '.join(non_recoverable_errors)}"
                )

            chunk_results.append(
                ChunkResult(
                    chunk_number=result.chunk_number,
                    success=False,
                    markdown_path=markdown_path,
                    error="; ".join(chunk_validation.errors),
                    validation=chunk_validation,
                )
            )
            continue

        chunk_results.append(
            ChunkResult(
                chunk_number=result.chunk_number,
                success=True,
                markdown_path=markdown_path,
                error=None,
                validation=chunk_validation,
            )
        )
        converted_chunks += 1

    if converted_chunks == 0:
        if converted_before_validation > 0:
            raise click.ClickException("All extracted chunks failed validation.")
        raise click.ClickException("Markdown conversion/cleanup failed for all extracted chunks.")

    failed_chunks = [*extraction_failure_results, *(item for item in chunk_results if not item.success)]

    warning_count = sum(len(item.validation.warnings) for item in chunk_results if item.validation is not None)
    logger.info(
        "Conversion summary: %s converted, %s chunks failed across extraction/conversion/validation, %s chunk validation warnings",
        converted_chunks,
        len(failed_chunks),
        warning_count,
    )

    try:
        merge_chunks(
            work_dir.markdown_dir,
            work_dir.merged_markdown(),
            split_result.chunk_count,
            logger=logger,
        )
    except MergeError as exc:
        raise click.ClickException(str(exc)) from exc

    log_heading_diagnostics(
        work_dir.markdown_dir,
        work_dir.merged_markdown(),
        logger=logger,
    )

    try:
        generate_toc(
            work_dir.merged_markdown(),
            work_dir.final_markdown(),
            toc_depth=cfg.toc_depth,
            logger=logger,
        )
    except TocError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        final_validation = validate_final_output(
            work_dir.final_markdown(),
            staged_input,
            [*extraction_failure_results, *chunk_results],
            max_empty_chunks=cfg.validation.max_empty_chunks,
            min_output_ratio=cfg.validation.min_output_ratio,
            logger=logger,
        )
    except FinalValidationError as exc:
        raise click.ClickException(str(exc)) from exc

    if not final_validation.valid:
        raise click.ClickException(" ; ".join(final_validation.errors))

    logger.info(
        "Final validation summary: warnings=%s toc_present=%s duplicate_boundaries=%s failed_chunks=%s",
        len(final_validation.warnings),
        final_validation.toc_present,
        final_validation.duplicate_boundary_count,
        final_validation.failed_chunk_count,
    )

    try:
        output_size_bytes = work_dir.final_markdown().stat().st_size
    except OSError:
        output_size_bytes = 0

    tables_found = sum(1 for _ in work_dir.tables_dir.glob("chunk-*-table-*.md"))
    total_warning_count = warning_count + len(final_validation.warnings)
    summary_context = RunSummaryContext(
        input_path=staged_input,
        input_size_bytes=validation.file_size_bytes,
        total_pages=validation.page_count,
        total_chunks=split_result.chunk_count,
        successful_chunks=converted_chunks,
        failed_chunks=tuple(failed_chunks),
        tables_found=tables_found,
        output_path=work_dir.final_markdown(),
        output_size_bytes=output_size_bytes,
        duration_seconds=perf_counter() - run_start,
        warning_count=total_warning_count,
    )
    summary_text = generate_run_summary(summary_context)
    click.echo(summary_text, err=True)
    try:
        append_run_summary(cfg.workdir / "run.log", summary_text)
    except RunSummaryError as exc:
        raise click.ClickException(str(exc)) from exc


def _version():
    from docdown import __version__
    return __version__


def _resolve_config_path(config_option: str | None) -> Path | None:
    if config_option is not None:
        return Path(config_option)
    default_config = Path("docdown.yaml")
    return default_config if default_config.exists() else None

