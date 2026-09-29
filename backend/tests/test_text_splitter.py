import unittest

from app.rag.text_splitter import MarkdownChunker


class MarkdownChunkerTest(unittest.TestCase):
    def test_preserves_heading_hierarchy_and_source_metadata(self) -> None:
        markdown = """# 복지제도

## 학자금 지원

### 지원 대상

임직원의 대학생 자녀를 대상으로 한다.
"""

        chunks = MarkdownChunker().split(
            markdown,
            metadata={"file_name": "복지제도.md", "category": "복지제도"},
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].metadata["heading"],
            "복지제도 > 학자금 지원 > 지원 대상",
        )
        self.assertEqual(chunks[0].metadata["file_name"], "복지제도.md")
        self.assertEqual(chunks[0].metadata["category"], "복지제도")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_splits_long_section_and_assigns_sequential_indexes(self) -> None:
        markdown = "# 긴 문서\n\n" + "\n\n".join(
            f"문단 {index} " + ("긴 내용 " * 30)
            for index in range(20)
        )

        chunks = MarkdownChunker(chunk_size=300, chunk_overlap=50).split(markdown)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.page_content) <= 300 for chunk in chunks))
        self.assertEqual(
            [chunk.metadata["chunk_index"] for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertTrue(all(chunk.metadata["heading"] == "긴 문서" for chunk in chunks))

    def test_keeps_short_table_and_list_together(self) -> None:
        markdown = """# 신청 안내

| 구분 | 금액 |
|---|---:|
| 결혼 | 500,000원 |
| 출산 | 300,000원 |

- 가족관계증명서
- 지급 신청서
"""

        chunks = MarkdownChunker().split(markdown)

        self.assertEqual(len(chunks), 1)
        self.assertIn("| 결혼 | 500,000원 |", chunks[0].page_content)
        self.assertIn("- 가족관계증명서", chunks[0].page_content)

    def test_removes_front_matter_and_html_comments(self) -> None:
        markdown = """---
title: 테스트 문서
source_pages: 3
---

# 본문

<!-- source_page: 1 -->

검색에 사용할 충분한 길이의 내용이다.
"""

        chunks = MarkdownChunker().split(markdown)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "검색에 사용할 충분한 길이의 내용이다.")
        self.assertNotIn("source_page", chunks[0].page_content)

    def test_returns_no_chunks_for_empty_document(self) -> None:
        chunks = MarkdownChunker().split("  \n<!-- source_page: 1 -->\n")

        self.assertEqual(chunks, [])

    def test_discards_non_searchable_short_fragments(self) -> None:
        markdown = """# 안내서

## 홈

홈

## 신청 방법

신청은 전자결재 시스템에서 진행한다.
"""

        chunks = MarkdownChunker().split(markdown)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["heading"], "안내서 > 신청 방법")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)


if __name__ == "__main__":
    unittest.main()
