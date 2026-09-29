import tempfile
import unittest
from pathlib import Path

from app.rag.document_loader import MarkdownDocumentLoader


class MarkdownDocumentLoaderTest(unittest.TestCase):
    def test_discovers_and_loads_markdown_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            category_directory = root / "복지제도"
            category_directory.mkdir()
            document_path = category_directory / "학자금.md"
            document_path.write_text("# 학자금\n\n지원 내용이다.", encoding="utf-8")
            (category_directory / "ignore.txt").write_text("제외", encoding="utf-8")

            loader = MarkdownDocumentLoader(root)
            files = loader.discover()
            document = loader.load(files[0])

            self.assertEqual(files, [document_path])
            self.assertEqual(document.metadata["file_path"], "복지제도/학자금.md")
            self.assertEqual(document.metadata["file_name"], "학자금.md")
            self.assertEqual(document.metadata["category"], "복지제도")
            self.assertIn("지원 내용", document.page_content)

    def test_checksum_changes_when_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            document_path = Path(temporary_directory) / "document.md"
            document_path.write_text("첫 번째 내용", encoding="utf-8")
            first_checksum = MarkdownDocumentLoader.checksum(document_path)

            document_path.write_text("두 번째 내용", encoding="utf-8")
            second_checksum = MarkdownDocumentLoader.checksum(document_path)

            self.assertEqual(len(first_checksum), 64)
            self.assertNotEqual(first_checksum, second_checksum)


if __name__ == "__main__":
    unittest.main()
