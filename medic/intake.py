from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Callable

import pymupdf

from .contracts import RawArtifact, SourceLocator
from .ledger import (
    AppendOnlyJsonlLedger,
    ContentAddressedStore,
    deterministic_id,
)
from .research_contracts import StrictFrozenModel


class InputKind(StrEnum):
    TEXT_DOCUMENT = "TEXT_DOCUMENT"
    PDF = "PDF"
    IMAGE = "IMAGE"


class ImageRole(StrEnum):
    DOCUMENT_IMAGE = "DOCUMENT_IMAGE"
    REFERENCE_IMAGE_SCREENSHOT = "REFERENCE_IMAGE_SCREENSHOT"


class IngestedFile(StrictFrozenModel):
    artifact: RawArtifact
    kind: InputKind
    image_role: ImageRole | None = None
    dicom_measurement_eligible: bool = False


class IntakeBlock(StrictFrozenModel):
    block_id: str
    artifact_id: str
    page_number: int | None = None
    text: str = ""
    source_locator: SourceLocator
    rendered_image_path: str | None = None
    untrusted_content: bool = True


class FileIntakeResult(StrictFrozenModel):
    intake_id: str
    case_id: str
    files: tuple[IngestedFile, ...]
    blocks: tuple[IntakeBlock, ...]


class FileIntakeService:
    SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
    SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}

    def __init__(
        self, store: ContentAddressedStore, ledger: AppendOnlyJsonlLedger
    ) -> None:
        self.store = store
        self.ledger = ledger

    def ingest_files(
        self,
        case_id: str,
        paths: list[Path],
        *,
        image_roles: dict[Path, ImageRole] | None = None,
        password_provider: Callable[[Path], str] | None = None,
    ) -> FileIntakeResult:
        image_roles = {Path(key): value for key, value in (image_roles or {}).items()}
        files: list[IngestedFile] = []
        blocks: list[IntakeBlock] = []
        for source_path in (Path(item) for item in paths):
            suffix = source_path.suffix.lower()
            if suffix not in self.SUPPORTED_TEXT | self.SUPPORTED_IMAGES | {".pdf"}:
                raise ValueError(f"unsupported input type: {source_path.suffix}")
            artifact = self.store.register_raw(case_id, source_path)
            raw_path = self.store.read_raw(artifact)
            if suffix in self.SUPPORTED_TEXT:
                files.append(IngestedFile(artifact=artifact, kind=InputKind.TEXT_DOCUMENT))
                text = raw_path.read_text(encoding="utf-8")
                blocks.append(self._text_block(artifact, text))
            elif suffix == ".pdf":
                files.append(IngestedFile(artifact=artifact, kind=InputKind.PDF))
                blocks.extend(self._pdf_blocks(artifact, raw_path, source_path, password_provider))
            else:
                role = image_roles.get(source_path, ImageRole.DOCUMENT_IMAGE)
                files.append(
                    IngestedFile(
                        artifact=artifact,
                        kind=InputKind.IMAGE,
                        image_role=role,
                        dicom_measurement_eligible=False,
                    )
                )
                blocks.append(self._image_block(artifact, raw_path))
        intake_id = deterministic_id(
            "file-intake",
            {"case_id": case_id, "version": 2,
             "files": [(item.artifact.artifact_id, item.image_role) for item in files]},
        )
        result = FileIntakeResult(
            intake_id=intake_id,
            case_id=case_id,
            files=tuple(files),
            blocks=tuple(blocks),
        )
        self.ledger.append_once(
            "REAL_FILES_INGESTED", intake_id, {"intake": result.model_dump(mode="json")}
        )
        return result

    @staticmethod
    def _text_block(artifact: RawArtifact, text: str) -> IntakeBlock:
        locator = SourceLocator(
            artifact_id=artifact.artifact_id,
            locator_type="text",
            locator="characters:0-end",
        )
        return IntakeBlock(
            block_id=deterministic_id("intake-block", {"artifact": artifact.artifact_id, "part": 0}),
            artifact_id=artifact.artifact_id,
            text=text,
            source_locator=locator,
        )

    def _pdf_blocks(
        self, artifact: RawArtifact, raw_path: Path, source_path: Path,
        password_provider: Callable[[Path], str] | None,
    ) -> list[IntakeBlock]:
        output: list[IntakeBlock] = []
        render_root = self.store.root / "derived" / "intake" / artifact.artifact_id
        with pymupdf.open(raw_path) as document:
            if document.needs_pass:
                if password_provider is None:
                    raise ValueError("PDF password required; use --pdf-password-prompt")
                if not document.authenticate(password_provider(source_path)):
                    raise ValueError("PDF authentication failed")
            for index, page in enumerate(document):
                text_blocks = [
                    block for block in page.get_text("blocks")
                    if len(block) >= 5 and str(block[4]).strip()
                ]
                if text_blocks:
                    for block_index, block in enumerate(text_blocks):
                        x0, y0, x1, y1, text = block[:5]
                        locator = SourceLocator(
                            artifact_id=artifact.artifact_id,
                            locator_type="pdf_bbox",
                            locator=(
                                f"page:{index + 1};bbox:"
                                f"{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}"
                            ),
                        )
                        output.append(
                            IntakeBlock(
                                block_id=deterministic_id(
                                    "intake-block",
                                    {"artifact": artifact.artifact_id, "page": index + 1,
                                     "block": block_index},
                                ),
                                artifact_id=artifact.artifact_id,
                                page_number=index + 1,
                                text=str(text).strip(),
                                source_locator=locator,
                            )
                        )
                    # A searchable footer does not prove that the scanned body was extracted.
                    if not page.get_image_info():
                        continue
                render_root.mkdir(parents=True, exist_ok=True)
                image_path = render_root / f"page-{index + 1}.png"
                if not image_path.exists():
                    page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(image_path)
                locator = SourceLocator(
                    artifact_id=artifact.artifact_id,
                    locator_type="pdf_page",
                    locator=f"page:{index + 1};whole-page-image",
                )
                output.append(
                    IntakeBlock(
                        block_id=deterministic_id(
                            "intake-block",
                            {"artifact": artifact.artifact_id, "page": index + 1},
                        ),
                        artifact_id=artifact.artifact_id,
                        page_number=index + 1,
                        source_locator=locator,
                        rendered_image_path=str(image_path),
                    )
                )
        return output
    @staticmethod
    def _image_block(artifact: RawArtifact, raw_path: Path) -> IntakeBlock:
        locator = SourceLocator(
            artifact_id=artifact.artifact_id,
            locator_type="image",
            locator="whole-image",
        )
        return IntakeBlock(
            block_id=deterministic_id("intake-block", {"artifact": artifact.artifact_id, "image": 1}),
            artifact_id=artifact.artifact_id,
            source_locator=locator,
            rendered_image_path=str(raw_path),
        )
