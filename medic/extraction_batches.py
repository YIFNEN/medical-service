from .intake import FileIntakeResult, IntakeBlock
from .ledger import deterministic_id


def extraction_batches(intake: FileIntakeResult) -> tuple[FileIntakeResult, ...]:
    """Stable file-local batches: at most four pages/images and 24k text characters."""
    output = []
    for file in intake.files:
        current: list[IntakeBlock] = []
        characters = images = 0
        pages: set[int | None] = set()

        def flush():
            if current:
                batch_id = deterministic_id("extraction-batch-v1", {
                    "case_id": intake.case_id, "file": file.model_dump(mode="json"),
                    "blocks": [block.model_dump(mode="json") for block in current],
                })
                output.append(FileIntakeResult(intake_id=batch_id, case_id=intake.case_id,
                                              files=(file,), blocks=tuple(current)))

        for source in intake.blocks:
            if source.artifact_id != file.artifact.artifact_id:
                continue
            for start in range(0, max(1, len(source.text)), 24000):
                block = source.model_copy(update={
                    "text": source.text[start:start + 24000],
                    "rendered_image_path": source.rendered_image_path if start == 0 else None,
                })
                image_count = int(bool(block.rendered_image_path))
                if current and (characters + len(block.text) > 24000 or images + image_count > 4
                                or len(pages | {block.page_number}) > 4 or len(current) >= 128):
                    flush()
                    current, pages, characters, images = [], set(), 0, 0
                current.append(block)
                pages.add(block.page_number)
                characters += len(block.text)
                images += image_count
        flush()
    return tuple(output)
