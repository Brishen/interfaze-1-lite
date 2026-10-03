"""Contract tests. No GPU, no network -- these are pure functions over fixtures."""

import ast

import pytest

from interfaze_lite.contracts import (
    OCRLine,
    OCRSection,
    OCRWord,
    Point,
    SpeakerTurn,
    VLMRegion,
    Word,
    attribute_speakers,
    group_by_speaker,
    iou,
    merge_lines_into_sections,
    pdf_render_dpi,
    rescale_page,
    stitch,
    to_bounds,
)

IMG_W, IMG_H = 1000, 800


def box(x1, y1, x2, y2):
    return to_bounds([x1, y1, x2, y2], src="xyxy", img_w=IMG_W, img_h=IMG_H)


class TestToBounds:
    def test_xyxy_produces_four_corners(self):
        b = to_bounds([10, 20, 110, 70], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert b.top_left == Point(10, 20)
        assert b.top_right == Point(110, 20)
        assert b.bottom_right == Point(110, 70)
        assert b.bottom_left == Point(10, 70)
        assert (b.width, b.height) == (100, 50)

    def test_norm1000_rescales_to_pixels(self):
        # Qwen-VL grounding grid. Half-width, quarter-height.
        b = to_bounds([0, 0, 500, 250], src="norm1000", img_w=1000, img_h=800)
        assert (b.width, b.height) == (500, 200)

    def test_norm01_rescales_to_pixels(self):
        b = to_bounds([0.1, 0.5, 0.2, 0.75], src="norm01", img_w=1000, img_h=800)
        assert (b.x1, b.y1, b.x2, b.y2) == (100, 400, 200, 600)

    def test_cxcywh_centres_the_box(self):
        b = to_bounds([100, 100, 40, 20], src="cxcywh", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (80, 90, 120, 110)

    def test_quad_takes_the_enclosing_axis_aligned_box(self):
        # A the line detector quad, rotated slightly.
        b = to_bounds([[10, 22], [110, 20], [112, 70], [12, 72]],
                      src="quad", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (10, 20, 112, 72)

    def test_reversed_corners_still_yield_a_positive_box(self):
        b = to_bounds([110, 70, 10, 20], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert (b.width, b.height) == (100, 50)

    def test_clamps_to_image(self):
        b = to_bounds([-50, -50, 2000, 2000], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert (b.x1, b.y1, b.x2, b.y2) == (0, 0, IMG_W, IMG_H)

    def test_width_always_equals_corner_difference(self):
        # Rounding once at the end keeps width consistent with the corners; rounding
        # separately can make them disagree by a pixel.
        b = to_bounds([10.6, 20.4, 110.4, 70.6], src="xyxy", img_w=IMG_W, img_h=IMG_H)
        assert b.width == b.x2 - b.x1
        assert b.height == b.y2 - b.y1

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            to_bounds([float("nan"), 0, 10, 10], src="xyxy", img_w=IMG_W, img_h=IMG_H)

    def test_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            to_bounds([1, 2, 3], src="xyxy", img_w=IMG_W, img_h=IMG_H)


class TestOCRComposition:
    def test_same_row_lines_merge_into_one_section(self):
        left = OCRLine("left column", box(10, 100, 200, 120), 0.9)
        right = OCRLine("right column", box(400, 100, 600, 120), 0.9)
        sections = merge_lines_into_sections([left, right])
        assert len(sections) == 1
        assert sections[0].text == "left column right column"

    def test_different_rows_stay_separate(self):
        top = OCRLine("first", box(10, 100, 200, 120), 0.9)
        bottom = OCRLine("second", box(10, 300, 200, 320), 0.9)
        assert len(merge_lines_into_sections([top, bottom])) == 2

    def test_differing_height_blocks_the_merge(self):
        # Same vertical band, but a heading is taller than body text.
        heading = OCRLine("HEADING", box(10, 100, 200, 140), 0.9)
        body = OCRLine("body", box(400, 100, 600, 120), 0.9)
        assert len(merge_lines_into_sections([heading, body])) == 2

    def test_merged_section_orders_lines_left_to_right(self):
        right = OCRLine("second", box(400, 100, 600, 120), 0.9)
        left = OCRLine("first", box(10, 100, 200, 120), 0.9)
        sections = merge_lines_into_sections([right, left])
        assert sections[0].text == "first second"


class TestStitch:
    def test_region_text_wins_detector_geometry_survives(self):
        region = VLMRegion(text="# Invoice", bounds=box(0, 0, 300, 50), category="title")
        line = OCRLine("Invoice", box(10, 10, 200, 40), 0.95,
                       words=[OCRWord("Invoice", box(10, 10, 200, 40), 0.95)])
        sections = stitch([region], [line])
        assert len(sections) == 1
        # Structure from the VLM...
        assert sections[0].text == "# Invoice"
        # ...geometry and confidence from the detector.
        assert sections[0].lines[0].average_confidence == 0.95
        assert sections[0].lines[0].bounds.x1 == 10

    def test_every_line_survives_and_takes_the_readers_words(self):
        """The detector owns boxes, the reader owns words -- and no box is dropped.

        Lines were matched to the reader's regions by position, and the reader's boxes
        are wrong: 62 of a form's 98 lines matched nothing and vanished.
        """
        region = VLMRegion(text="Invoice total 42.00", bounds=box(0, 0, 10, 10))
        matched = OCRLine("Invoicetotal 42.00", box(10, 10, 200, 40), 0.9)   # spaces dropped
        orphan = OCRLine("footnote", box(10, 700, 200, 730), 0.6)          # not in the reader
        [section] = stitch([region], [matched, orphan])
        assert [ln.text for ln in section.lines] == ["Invoice total 42.00", "footnote"]
        assert [ln.bounds.y1 for ln in section.lines] == [10, 700]
        assert section.text == "Invoice total 42.00"

    def test_a_number_inside_a_longer_one_is_not_a_match(self):
        """"150.02" occurs inside the barcode 005150025499; it is not that barcode."""
        region = VLMRegion(text="PEANUT BUTTR 005150025499 F 5.44 CASH TEND 150.02",
                           bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("150.02", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "150.02"

    def test_a_label_the_reader_has_once_goes_to_one_line(self):
        region = VLMRegion(text="Dependent 1 First name", bounds=box(0, 0, 10, 10))
        lines = [OCRLine("Dependent 1", box(10, 10, 100, 40), 0.9),
                 OCRLine("Dependent 2", box(110, 10, 200, 40), 0.9)]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == ["Dependent 1", "Dependent 2"]

    def test_another_case_is_another_occurrence(self):
        region = VLMRegion(text="Dr. Aaron Patel, MD", bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("DR. AARON PATEL", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "DR. AARON PATEL"  # the stamp, not the provider line

    def test_a_word_broken_across_lines_is_split_where_the_line_breaks(self):
        region = VLMRegion(text="leads a suite of deterministic developer-task benchmarks",
                           bounds=box(0, 0, 10, 10))
        lines = [OCRLine("leads a suite of de-", box(10, 10, 300, 40), 0.9),
                 OCRLine("terministic developer-task benchmarks", box(10, 50, 300, 80), 0.9)]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == [
            "leads a suite of de-", "terministic developer-task benchmarks"]

    def test_reader_text_no_line_found_is_still_the_pages_text(self):
        region = VLMRegion(text="orphan heading", bounds=box(0, 0, 50, 20))
        far = OCRLine("elsewhere", box(900, 700, 990, 740), 0.9)
        [section] = stitch([region], [far])
        assert section.text == "orphan heading"
        assert [ln.text for ln in section.lines] == ["elsewhere"]

    def test_lines_are_found_by_what_they_say_not_where_the_reader_put_them(self):
        """The reader's x-coordinates come back roughly halved; its y-coordinates hold.

        Matched on box containment, the right column's lines fell inside no region and
        a two-column page lost half its geometry.
        """
        left = VLMRegion(text="Many production workloads are not open ended",
                         bounds=box(75, 100, 290, 160))
        right = VLMRegion(text="A common remedy is to route between models",
                          bounds=box(290, 100, 500, 160))
        lines = [OCRLine("Many production workloads", box(150, 100, 430, 125), 0.9),
                 OCRLine("are not open ended", box(150, 130, 430, 155), 0.9),
                 OCRLine("A common remedy is", box(430, 100, 850, 125), 0.9),
                 OCRLine("toroute between models", box(430, 130, 850, 155), 0.9)]
        [section] = stitch([left, right], lines)
        assert [ln.text for ln in section.lines] == [
            "Many production workloads", "A common remedy is",
            "are not open ended", "to route between models"]
        assert section.lines[1].bounds.x1 == 430  # the detector's box, not the reader's

    def test_each_line_gets_its_own_stretch_of_the_paragraph(self):
        """Every line carrying the whole paragraph multiplied the page's text by its lines."""
        region = VLMRegion(text="Interfaze fuses task-specific encoders into a decoder "
                                "through a shared embedding space, so one pass does it.",
                           bounds=box(0, 0, 400, 100))
        lines = [OCRLine("Interfazefuses task-specific encoders", box(0, 0, 400, 30), 0.9),
                 OCRLine("into adecoder through a shared", box(0, 35, 400, 65), 0.9),
                 OCRLine("embedding space, so one pass does it.", box(0, 70, 400, 100), 0.9)]
        [section] = stitch([region], lines)
        assert [ln.text for ln in section.lines] == [
            "Interfaze fuses task-specific encoders",
            "into a decoder through a shared",
            "embedding space, so one pass does it."]

    def test_a_capital_dotted_i_does_not_shift_the_text(self):
        """'İ'.lower() is two characters. Lowercased as-is, every İ pushed the searched
        text one place past its positions, and a line found near the end of a Turkish
        table indexed past them: the page's OCR failed with IndexError."""
        region = VLMRegion(text="İSTANBUL İZMİR İÇ TİCARET Toplam 2024 1.250.000",
                           bounds=box(0, 0, 10, 10))
        [section] = stitch([region], [OCRLine("Toplam 2024 1.250.000", box(10, 10, 200, 40), 0.9)])
        assert section.lines[0].text == "Toplam 2024 1.250.000"

    def test_iou_is_zero_for_disjoint_boxes(self):
        assert iou(box(0, 0, 10, 10), box(100, 100, 110, 110)) == 0.0


class TestPdfRenderDpi:
    BUDGET, SMALL = 4_194_304, 2_097_152

    def dpi(self, width_pt, height_pt):
        return pdf_render_dpi(width_pt, height_pt, dpi=200, max_pixels=self.BUDGET, small_pixels=self.SMALL)

    def test_a_standard_page_keeps_the_default(self):
        assert self.dpi(595, 842) == 200  # A4
        assert self.dpi(612, 792) == 200  # Letter

    def test_a_small_page_is_read_at_the_pixel_budget(self):
        """A 4.3 x 2.8 in page of tiny print, scanned at 2592x1703, was rendered at 200
        DPI -- 864x567, 11% of the scan's pixels -- and the reader invented sentences.
        At the budget it passed 38 of 39 olmOCR tests instead of 29."""
        dpi = self.dpi(311.04, 204.36)
        assert 560 <= dpi <= 600
        assert (311.04 / 72 * dpi) * (204.36 / 72 * dpi) <= self.BUDGET

    def test_an_oversized_page_is_rendered_down_to_the_budget(self):
        dpi = self.dpi(1200, 1800)
        assert dpi < 200
        assert (1200 / 72 * dpi) * (1800 / 72 * dpi) <= self.BUDGET

    def test_never_below_72_dpi(self):
        assert self.dpi(20000, 30000) == 72

    def test_an_empty_page_keeps_the_default(self):
        assert self.dpi(0, 0) == 200


class TestRescalePage:
    def test_every_box_moves_into_the_callers_frame(self):
        b = {"top_left": {"x": 100, "y": 200}, "top_right": {"x": 300, "y": 200},
             "bottom_right": {"x": 300, "y": 250}, "bottom_left": {"x": 100, "y": 250},
             "width": 200, "height": 50}
        page = {"width": 1700, "height": 2200, "layout": [{"type": "text", "bounds": b}],
                "sections": [{"page": 1, "width": 1700, "height": 2200, "lines": [
                    {"text": "a", "bounds": b, "words": [{"text": "a", "bounds": b}]}]}]}
        out = rescale_page(page, 1224, 1584)
        assert (out["width"], out["height"]) == (1224, 1584)
        assert (out["sections"][0]["width"], out["sections"][0]["height"]) == (1224, 1584)
        for moved in (out["layout"][0]["bounds"], out["sections"][0]["lines"][0]["bounds"],
                      out["sections"][0]["lines"][0]["words"][0]["bounds"]):
            assert moved["top_left"] == {"x": 72, "y": 144}
            assert moved["bottom_right"] == {"x": 216, "y": 180}
            assert (moved["width"], moved["height"]) == (144, 36)

    def test_same_frame_is_untouched(self):
        page = {"width": 720, "height": 960, "sections": []}
        assert rescale_page(page, 720, 960) is page


class TestSpeakerAttribution:
    def test_max_overlap_beats_midpoint_on_interleaved_turns(self):
        """The case that distinguishes the correct algorithm from the common wrong one.

        Turns interleave, and a long word spans several. Total overlap favours spk_1
        (1.0 + 2.0 = 3.0) over spk_0 (0.5 + 1.0 = 1.5). But the word's midpoint is
        5.75, which lands inside a spk_0 turn -- so midpoint assignment would answer
        spk_0. NeMo's forward-only pointer scan fails here too.
        """
        turns = [
            SpeakerTurn("spk_0", 0.0, 4.0),
            SpeakerTurn("spk_1", 4.0, 5.0),
            SpeakerTurn("spk_0", 5.0, 6.0),
            SpeakerTurn("spk_1", 6.0, 9.0),
        ]
        [word] = attribute_speakers([Word("mmhmm", 3.5, 8.0)], turns)
        assert word.speaker == "spk_1"

        midpoint = (3.5 + 8.0) / 2
        containing = [t.speaker for t in turns if t.start <= midpoint <= t.end]
        assert containing == ["spk_0"], "fixture must actually discriminate the two rules"

    def test_simple_word_lands_in_its_turn(self):
        turns = [SpeakerTurn("spk_0", 0.0, 5.0), SpeakerTurn("spk_1", 5.0, 10.0)]
        words = attribute_speakers(
            [Word("hello", 1.0, 2.0), Word("world", 6.0, 7.0)], turns)
        assert [w.speaker for w in words] == ["spk_0", "spk_1"]

    def test_no_overlap_leaves_speaker_none_by_default(self):
        turns = [SpeakerTurn("spk_0", 10.0, 20.0)]
        [word] = attribute_speakers([Word("cough", 0.0, 1.0)], turns)
        assert word.speaker is None

    def test_fill_nearest_assigns_the_closest_turn(self):
        turns = [SpeakerTurn("spk_0", 10.0, 20.0), SpeakerTurn("spk_1", 100.0, 110.0)]
        [word] = attribute_speakers([Word("cough", 0.0, 1.0)], turns, fill_nearest=True)
        assert word.speaker == "spk_0"

    def test_long_turn_is_not_penalised(self):
        """IoU would divide by turn duration and pick the short turn. Overlap must not."""
        turns = [SpeakerTurn("spk_long", 0.0, 600.0), SpeakerTurn("spk_short", 5.0, 5.2)]
        [word] = attribute_speakers([Word("yes", 4.0, 5.1)], turns)
        assert word.speaker == "spk_long"


class TestGroupBySpeaker:
    def test_contiguous_words_collapse_into_one_chunk(self):
        words = [
            Word("the", 0.0, 0.5, "spk_0"),
            Word("total", 0.5, 1.0, "spk_0"),
            Word("forty", 1.2, 1.8, "spk_1"),
        ]
        chunks = group_by_speaker(words)
        assert len(chunks) == 2
        assert chunks[0] == {"speaker": "spk_0", "text": "the total", "timestamp": [0.0, 1.0]}
        assert chunks[1]["speaker"] == "spk_1"

    def test_speaker_alternation_splits_chunks(self):
        words = [
            Word("a", 0.0, 0.1, "spk_0"),
            Word("b", 0.1, 0.2, "spk_1"),
            Word("c", 0.2, 0.3, "spk_0"),
        ]
        assert len(group_by_speaker(words)) == 3


class TestChandraHTML:
    """Chandra 2 returns HTML carrying data-bbox; the parser must survive real output."""

    def _parse(self, html, w=1600, h=1200):
        from interfaze_lite.contracts import parse_vlm_regions

        class Img:
            width, height = w, h
        return parse_vlm_regions(html, Img())

    def test_extracts_text_and_scales_normalised_boxes(self):
        html = ('<h1 data-bbox="100 50 500 120">INVOICE 2026-0042</h1>'
                '<p data-bbox="100 200 700 260">TOTAL 166.20</p>')
        regions = self._parse(html)
        assert [r.text for r in regions] == ["INVOICE 2026-0042", "TOTAL 166.20"]
        # 0-1000 grid against a 1600x1200 page
        assert regions[0].bounds.x1 == 160
        assert regions[0].bounds.y1 == 60

    def test_absolute_pixels_are_left_alone(self):
        html = '<p data-bbox="1200 900 1500 1000">footer</p>'
        [region] = self._parse(html)
        assert region.bounds.x1 == 1200 and region.bounds.y1 == 900

    def test_no_bbox_yields_no_regions(self):
        assert self._parse("<p>plain html, no boxes</p>") == []

    def test_empty_regions_are_dropped(self):
        html = '<div data-bbox="0 0 100 100"></div><p data-bbox="0 200 100 300">real</p>'
        assert [r.text for r in self._parse(html)] == ["real"]


class TestDocumentKinds:
    """PDF and DOCX must be detected by CONTENT: these arrive as ref-N temp files whose
    suffix came from a guessed MIME type."""

    def _kind(self, blob):
        from interfaze_lite.contracts import document_kind
        return document_kind(blob)

    def test_pdf_detected_by_magic_bytes(self):
        assert self._kind(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n") == "pdf"

    def test_docx_detected_by_zip_plus_word_entry(self):
        assert self._kind(b"PK\x03\x04" + b"\x00" * 30 + b"word/document.xml") == "docx"

    def test_plain_zip_is_not_docx(self):
        assert self._kind(b"PK\x03\x04" + b"\x00" * 30 + b"xl/workbook.xml") == "image"

    def test_png_is_an_image(self):
        assert self._kind(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "image"


class TestSectionPageFrame:
    """Bounds are page-local; each section names its page and that page's frame.

    interfaze never translates coordinates across pages (ocr/vocr.ts:594-605). This code
    previously stacked pages onto one canvas by offsetting each page's y, which silently
    moved every box on every page after the first.
    """

    def test_backfill_stamps_frame_without_moving_bounds(self):
        from interfaze_lite.contracts import backfill_sections

        section = {"text": "x", "lines": [{"text": "x", "bounds": {
            "top_left": {"x": 10, "y": 20}, "top_right": {"x": 90, "y": 20},
            "bottom_right": {"x": 90, "y": 50}, "bottom_left": {"x": 10, "y": 50},
            "width": 80, "height": 30}}]}
        [out] = backfill_sections([section], page=3, width=1240, height=1754)

        assert (out["page"], out["width"], out["height"]) == (3, 1240, 1754)
        # page 3's coordinates are still page-local
        assert out["lines"][0]["bounds"]["top_left"] == {"x": 10, "y": 20}

    def test_backfill_does_not_mutate_the_input(self):
        from interfaze_lite.contracts import backfill_sections

        section = {"text": "x", "lines": []}
        backfill_sections([section], page=2, width=100, height=200)
        assert "page" not in section


class TestFileContentPart:
    """OpenAI's generic `file` part. interfaze's own benchmark harness sends audio this
    way, so a client using it was silently getting answers invented from the prompt."""

    def _extract(self, part):
        from interfaze_lite.filerefs import extract_from_messages
        return extract_from_messages([{"role": "user", "content": [
            {"type": "text", "text": "transcribe"}, part]}])

    def test_file_url_becomes_a_ref(self):
        refs, msgs = self._extract({"type": "file", "file": {
            "filename": "call.wav", "file_url": "https://x.test/call.wav"}})
        assert refs.resolve("ref-0") == "https://x.test/call.wav"
        assert "ref-0" in msgs[0]["content"][1]["text"]

    def test_data_uri_is_materialised(self):
        refs, _ = self._extract({"type": "file", "file": {
            "filename": "a.txt", "file_data": "data:text/plain;base64,aGVsbG8="}})
        assert refs.resolve("ref-0").endswith((".txt", ".bin")) or refs.resolve("ref-0")

    def test_bare_base64_uses_the_filename_for_mime(self):
        refs, _ = self._extract({"type": "file", "file": {
            "filename": "clip.wav", "file_data": "aGVsbG8="}})
        assert refs.refs["ref-0"].mime == "audio/x-wav" or "wav" in refs.refs["ref-0"].mime

    def test_empty_payload_is_left_alone_not_faked(self):
        refs, msgs = self._extract({"type": "file", "file": {"filename": "x.pdf"}})
        assert refs.refs == {}
        assert msgs[0]["content"][1]["type"] == "file"


class TestOneSectionPerPage:
    """interfaze consumers index sections[0] as the page and derive per-page height
    from len(sections). More than one section per page silently misplaces every box."""

    def test_regions_collapse_into_a_single_section(self):
        from interfaze_lite.contracts import collapse_to_page

        a = OCRSection(text="Heading", lines=[OCRLine("Heading", box(10, 10, 200, 40), 0.9)])
        b = OCRSection(text="Body", lines=[OCRLine("Body", box(10, 60, 300, 90), 0.8)])
        page = collapse_to_page([a, b])
        assert page.text == "Heading\nBody"
        assert len(page.lines) == 2

    def test_lines_are_ordered_top_then_left(self):
        from interfaze_lite.contracts import collapse_to_page

        lower = OCRSection(text="b", lines=[OCRLine("b", box(10, 500, 100, 530), 0.9)])
        upper_right = OCRSection(text="a2", lines=[OCRLine("a2", box(400, 10, 500, 40), 0.9)])
        upper_left = OCRSection(text="a1", lines=[OCRLine("a1", box(10, 10, 100, 40), 0.9)])
        page = collapse_to_page([lower, upper_right, upper_left])
        assert [line.text for line in page.lines] == ["a1", "a2", "b"]

    def test_empty_region_text_is_not_joined_in(self):
        from interfaze_lite.contracts import collapse_to_page

        page = collapse_to_page([
            OCRSection(text="real", lines=[]),
            OCRSection(text="   ", lines=[]),
        ])
        assert page.text == "real"


class TestScanEnhancement:
    def _img(self, lo, hi):
        from PIL import Image
        img = Image.new("RGB", (40, 40), (hi, hi, hi))
        for x in range(0, 40, 2):
            for y in range(40):
                img.putpixel((x, y), (lo, lo, lo))
        return img

    def test_faded_scan_is_stretched(self):
        from interfaze_lite.contracts import enhance_scan

        out = enhance_scan(self._img(90, 170))
        levels = out.convert("L").getextrema()
        assert levels[1] - levels[0] > 150, f"contrast not stretched: {levels}"

    def test_clean_page_is_untouched(self):
        from interfaze_lite.contracts import enhance_scan

        original = self._img(0, 255)
        assert enhance_scan(original) is original


class TestDetectorRescale:
    def test_quad_scales_into_the_callers_frame(self):
        from interfaze_lite.contracts import to_bounds

        # A box found on a half-size copy must land in the same place on the full page.
        quad_small = [[100, 50], [200, 50], [200, 80], [100, 80]]
        sx = sy = 2.0
        scaled = [[x * sx, y * sy] for x, y in quad_small]
        bounds = to_bounds(scaled, src="quad", img_w=1600, img_h=1200)
        assert bounds.top_left.x == 200
        assert bounds.top_left.y == 100
        assert bounds.bottom_right.x == 400
        assert bounds.bottom_right.y == 160


class TestReadDocumentShape:
    """Both Chandra paths must return (regions, markdown).

    They did not: the vLLM path returned a tuple and the in-process path a bare list,
    so `regions, markdown = self._read_document(img)` raised on any page with a number
    of regions other than two -- and silently mis-bound on exactly two. Modal sets
    OCR_VLM_URL so only the Docker image and the HF repo hit it, which is precisely
    where nobody was looking.
    """

    def test_both_reader_branches_return_a_pair(self):
        import ast
        import pathlib

        source = pathlib.Path("src/interfaze_lite/modeling_interfaze_lite.py").read_text()
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_read_document_html")
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value]
        assert returns, "expected return statements"
        for node in returns:
            assert isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2, (
                f"_read_document_html returns a non-pair at line {node.lineno}")


@pytest.mark.skipif(not __import__("pathlib").Path("scripts_sync_hf.sh").exists(),
                    reason="Hugging Face packaging lives with the model repo, not here")
class TestHuggingFacePackaging:
    """The generated HF repo must be self-contained and agree with the code.

    It was neither: `lam_common` was imported but never copied, so the detection path
    raised ModuleNotFoundError; and config.json pinned an older ASR checkpoint which
    wins over the code default, handing every HF user the model that OOMed.
    """

    def test_every_sibling_module_reaches_the_user(self):
        """trust_remote_code downloads a sibling only if a `from .x import` line names it.

        Its own pattern is used here. `from . import contracts` does not match it, so
        contracts.py, grounding.py and the rest never left the Hub: every
        from_pretrained ended in ImportError.
        """
        import pathlib
        import re

        src = pathlib.Path("src/interfaze_lite")
        script = pathlib.Path("scripts_sync_hf.sh").read_text()
        todo, shipped = ["modeling_interfaze_lite"], set()
        while todo:
            name = todo.pop()
            if name in shipped:
                continue
            shipped.add(name)
            assert f"src/interfaze_lite/{name}.py" in script, f"scripts_sync_hf.sh does not copy {name}.py"
            text = (src / f"{name}.py").read_text()
            found = set(re.findall(r"^\s*import\s+\.(\S+)\s*$", text, re.M))
            found |= set(re.findall(r"^\s*from\s+\.(\S+)\s+import", text, re.M))
            for sibling in re.findall(r"^\s*from \. import ([\w, ]+)", text, re.M):
                for dep in re.split(r"[, ]+", sibling):
                    assert not dep or dep in found, (
                        f"{name}.py imports {dep} only as `from . import {dep}`, which "
                        "trust_remote_code does not ship")
            todo += [dep for dep in found if (src / f"{dep}.py").exists()]
        assert {"contracts", "grounding", "agent", "prompts"} <= shipped

    def test_the_published_config_is_bundled_and_names_no_component(self):
        """Every component loads from its own folder in the repo, and none is named.

        Upstream ids stay out of config.json: the bundle's folders are what load, and
        naming the checkpoints would say what the model is made of.
        """
        import json
        import pathlib

        template = json.loads(pathlib.Path("hf_config.json.tmpl").read_text())
        assert template["bundle_weights"] is True
        assert not template.get("components")


class TestSourceIntegrity:
    """Every shipped module must compile, and carry no conflict markers.

    A failed `git stash pop` once left conflict markers inside a parenthesised
    expression in brain.py. The module was a SyntaxError, yet the suite stayed green
    because pytest loaded a stale __pycache__ entry -- so a file that could not be
    imported was deployed with every test passing.
    """

    def _shipped(self):
        import pathlib

        return [p for p in [*pathlib.Path("src").rglob("*.py"), pathlib.Path("modal_app.py")] if p.exists()]

    def test_no_conflict_markers(self):
        for path in self._shipped():
            text = path.read_text()
            for marker in ("<" * 7, ">" * 7):
                assert marker not in text, f"{path} carries a conflict marker"

    def test_every_module_compiles(self):
        import ast

        for path in self._shipped():
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                raise AssertionError(f"{path} does not parse: {exc}") from exc


class TestErrorNormalisation:
    """Upstream error text is restated in capability terms.

    The identifiers are passed in by the caller from its own configuration, so this
    test invents its own rather than naming anything real.
    """

    VENDORS = ("acme/speech-xl-v3", "globex/segment-base", "initech/reader-2")

    def test_repo_ids_are_replaced(self):
        from interfaze_lite.contracts import normalise_error

        for vendor in self.VENDORS:
            out = normalise_error(f"{vendor} failed to load", self.VENDORS)
            assert "<component>" in out
            assert vendor not in out

    def test_bare_words_are_replaced(self):
        from interfaze_lite.contracts import normalise_error

        out = normalise_error("Initech reader blew up in native code", self.VENDORS)
        assert "initech" not in out.lower()

    def test_ordinary_errors_are_untouched(self):
        from interfaze_lite.contracts import normalise_error

        text = "request timed out after 600s"
        assert normalise_error(text, self.VENDORS) == text

    def test_no_identifiers_without_configuration(self):
        from interfaze_lite.contracts import normalise_error

        text = "acme/speech-xl-v3 failed"
        assert normalise_error(text, ()) == text

    def test_health_payloads_name_capabilities(self):
        import pathlib

        for name in ("perception.py", "diarize.py"):
            source = pathlib.Path(f"src/interfaze_lite/services/{name}").read_text()
            health = source.split("def health(")[1].split("\n\n\n")[0]
            assert "DEFAULT_PIPELINE" not in health
            assert "capabilit" in health.lower() or "CAPABILITY" in health


class TestLayoutTextJoin:
    def _line(self, text, x1, y1, x2, y2):
        from interfaze_lite.contracts import Bounds, OCRLine, Point

        return OCRLine(
            text=text,
            bounds=Bounds(top_left=Point(x1, y1), top_right=Point(x2, y1),
                          bottom_right=Point(x2, y2), bottom_left=Point(x1, y2),
                          width=x2 - x1, height=y2 - y1),
            average_confidence=0.9, words=[])

    def _block(self, kind, x1, y1, x2, y2):
        return {"type": kind, "score": 0.9, "bounds": {
            "top_left": {"x": x1, "y": y1}, "top_right": {"x": x2, "y": y1},
            "bottom_right": {"x": x2, "y": y2}, "bottom_left": {"x": x1, "y": y2},
            "width": x2 - x1, "height": y2 - y1}}

    def test_each_block_gets_the_text_inside_it(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("Annual Report", 10, 10, 200, 40),
                 self._line("Revenue rose", 10, 100, 300, 130),
                 self._line("in every region.", 10, 140, 300, 170)]
        blocks = [self._block("title", 0, 0, 400, 50),
                  self._block("paragraph", 0, 90, 400, 180)]
        out = attach_text_to_layout(blocks, lines)
        assert out[0]["type"] == "title" and out[0]["text"] == "Annual Report"
        assert out[1]["type"] == "paragraph"
        assert out[1]["text"] == "Revenue rose in every region."
        assert out[1]["line_count"] == 2

    def test_text_outside_every_block_is_kept(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("boxed", 10, 10, 100, 40),
                 self._line("orphan footnote", 10, 900, 300, 930)]
        out = attach_text_to_layout([self._block("title", 0, 0, 200, 50)], lines)
        assert out[-1]["type"] == "unclassified"
        assert out[-1]["text"] == "orphan footnote"

    def test_a_line_is_claimed_once(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("shared", 10, 10, 100, 40)]
        blocks = [self._block("title", 0, 0, 200, 50),
                  self._block("paragraph", 0, 0, 200, 50)]
        out = attach_text_to_layout(blocks, lines)
        assert [b["line_count"] for b in out] == [1, 0]

    def _region(self, text, x1, y1, x2, y2):
        from interfaze_lite.contracts import VLMRegion
        return VLMRegion(text=text, bounds=self._line("", x1, y1, x2, y2).bounds)

    def test_the_readers_text_wins_over_the_line_detectors(self):
        # The detector reads "evaluatereasoning" and "StructE- val"; the reader does not.
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("models evaluatereasoning, StructE-", 10, 100, 300, 130),
                 self._line("val and more", 10, 140, 300, 170)]
        regions = [self._region("models evaluate reasoning, StructEval and more", 5, 95, 305, 175)]
        out = attach_text_to_layout([self._block("text", 0, 90, 400, 180)], lines, regions=regions)
        assert out[0]["text"] == "models evaluate reasoning, StructEval and more"
        assert out[0]["line_count"] == 2
        assert len(out) == 1

    def test_a_line_the_reader_missed_keeps_the_detectors_reading(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("Figure 1", 10, 10, 100, 40)]
        regions = [self._region("A paragraph.", 10, 100, 300, 170)]
        out = attach_text_to_layout([self._block("figure_title", 0, 0, 200, 50)], lines, regions=regions)
        assert [b["text"] for b in out] == ["Figure 1"]

    def test_the_readers_boxes_place_nothing(self):
        # A landscape photo of a form came back with the reader's boxes transposed:
        # the letterhead at x=509, y=44-1056. Blocks still get their own text.
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("WESTVIEW ORTHOPAEDIC", 640, 40, 1270, 90),
                 self._line("Patient Name: Sarah K. Miller", 400, 280, 870, 325)]
        regions = [self._region("WESTVIEW ORTHOPAEDIC CLINIC", 509, 44, 734, 1056),
                   self._region("Patient Name: Sarah K. Miller DOB: 1995-11-03", 379, 616, 734, 1100)]
        out = attach_text_to_layout([self._block("doc_title", 600, 30, 1300, 100),
                                     self._block("text", 390, 270, 900, 330)], lines, regions=regions)
        assert [b["text"] for b in out] == ["WESTVIEW ORTHOPAEDIC", "Patient Name: Sarah K. Miller"]

    def test_side_by_side_columns_keep_their_own_lines(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("left column about apples", 10, 100, 190, 130),
                 self._line("right column about rivers", 210, 100, 390, 130)]
        blocks = [self._block("text", 0, 90, 200, 140), self._block("text", 200, 90, 400, 140)]
        regions = [self._region("right column about rivers.", 20, 95, 150, 135),
                   self._region("left column about apples.", 10, 95, 180, 135)]
        out = attach_text_to_layout(blocks, lines, regions=regions)
        assert out[0]["text"] == "left column about apples."
        assert out[1]["text"] == "right column about rivers."

    def test_a_list_the_reader_kept_whole_is_split_across_its_item_blocks(self):
        # One List-Group region over three item blocks: each block gets its own item.
        from interfaze_lite.contracts import VLMRegion, attach_text_to_layout

        lines = [self._line("1. firstitem", 10, 100, 300, 130),
                 self._line("2. second item", 10, 150, 300, 180),
                 self._line("3. third item", 10, 200, 300, 230)]
        blocks = [self._block("text", 0, 95, 400, 135), self._block("text", 0, 145, 400, 185),
                  self._block("text", 0, 195, 400, 235)]
        markup = "<ol><li>1. first item</li><li>2. second item</li><li>3. third item</li></ol>"
        region = VLMRegion(text="1. first item 2. second item 3. third item", markup=markup,
                           category="List-Group", bounds=self._line("", 10, 95, 300, 235).bounds)
        out = attach_text_to_layout(blocks, lines, regions=[region])
        assert [b["text"] for b in out] == ["1. first item", "2. second item", "3. third item"]

    def test_a_list_that_does_not_line_up_is_not_repeated_as_unclassified(self):
        from interfaze_lite.contracts import VLMRegion, attach_text_to_layout

        lines = [self._line("1. first", 10, 100, 300, 130), self._line("2. second", 10, 150, 300, 180)]
        blocks = [self._block("text", 0, 95, 400, 135), self._block("text", 0, 145, 400, 185)]
        region = VLMRegion(text="1. first 2. second 3. extra", markup="<ol><li>a</li><li>b</li><li>c</li></ol>",
                           bounds=self._line("", 10, 95, 300, 185).bounds)
        out = attach_text_to_layout(blocks, lines, regions=[region])
        assert [b["text"] for b in out] == ["1. first", "2. second"]
        assert len(out) == 2

    def test_tables_come_back_as_the_reader_produced_them(self):
        from interfaze_lite.contracts import VLMRegion, attach_text_to_layout

        table = "<table><tr><td>Metric</td><td>Range</td></tr></table>"
        region = VLMRegion(text="Metric Range", markup=table, category="Table",
                           bounds=self._line("", 700, 5, 900, 20).bounds)
        lines = [self._line("Metric Range", 10, 100, 300, 130)]
        out = attach_text_to_layout([self._block("table", 0, 90, 400, 180)], lines, regions=[region])
        assert out[0]["text"] == table

    def test_unboxed_reader_text_is_kept_once(self):
        from interfaze_lite.contracts import attach_text_to_layout

        lines = [self._line("orphan footnote", 10, 900, 300, 930)]
        regions = [self._region("orphan footnote", 5, 895, 305, 935)]
        out = attach_text_to_layout([self._block("title", 0, 0, 200, 50)], lines, regions=regions)
        assert out[-1]["type"] == "unclassified"
        assert out[-1]["text"] == "orphan footnote"
        assert len(out) == 2


class TestReaderSelection:
    """The reader is chosen by configuration, never by the component's name.

    It used to branch on a substring of the configured repo id, so renaming that
    identifier silently routed every page to the other reader -- which then failed,
    and OCR returned no text at all with the layout list empty.
    """

    def test_branch_does_not_sniff_the_component_name(self):
        import inspect
        import pathlib

        source = pathlib.Path(
            "src/interfaze_lite/modeling_interfaze_lite.py").read_text()
        body = source.split("def _read_document(")[1].split("\n    def ")[0]
        assert "ocr_vlm_format" in body, "reader choice should read the format setting"
        assert 'component("ocr_vlm")' not in body, (
            "reader choice must not depend on the component identifier")
        assert inspect  # keep the import meaningful if the check above changes


class TestEveryReadPathClassifies:
    """Both read paths must hand layout to `_assemble`.

    The single-image path ran layout from the start; the PDF path never called the
    detector at all, and because `_assemble` defaults the status to "not run" a
    fifteen-page document reported an untyped layout that looked identical to a page
    with no regions. Checked at the source level because the difference between the
    two paths is which arguments a call site passes, which no fixture exercises.
    """

    @staticmethod
    def _assemble_calls(func_name: str) -> list[ast.Call]:
        import pathlib
        tree = ast.parse(pathlib.Path(
            "src/interfaze_lite/modeling_interfaze_lite.py").read_text())
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func_name)
        return [n for n in ast.walk(func)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_assemble"]

    @pytest.mark.parametrize("path", ["_ocr_pdf", "_ocr_image"])
    def test_layout_is_passed(self, path):
        calls = self._assemble_calls(path)
        assert calls, f"{path} makes no _assemble call"
        for call in calls:
            passed = {kw.arg for kw in call.keywords}
            assert "layout" in passed, f"{path} drops layout"
            assert "layout_status" in passed, f"{path} drops layout_status"


class TestIdentifiersAreIntact:
    """No string literal that is used as an identifier may contain prose.

    A repo id, package name or env-var default is consumed verbatim by another
    system. One that reads as a phrase was edited by something matching on text
    rather than on syntax, and the damage is invisible until the call fails far away:
    a mangled diarizer id surfaced only as a hub validation error inside a subprocess,
    which reached the caller as a bare 500.
    """

    @pytest.mark.parametrize("relative", [
        "src/interfaze_lite/config.py",
        "src/interfaze_lite/services/diarize.py",
        "src/interfaze_lite/configuration_interfaze_lite.py",
        "src/interfaze_lite/modeling_interfaze_lite.py",
        "modal_app.py",
    ])
    def test_no_prose_in_identifier_literals(self, relative):
        import pathlib
        if not pathlib.Path(relative).exists():
            pytest.skip(f"{relative} is not part of this checkout")
        tree = ast.parse(pathlib.Path(relative).read_text())
        bad = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            # Exactly `owner/name`, with a space in it. One slash is what separates a
            # repo id from a URL or a user-agent, both of which have several and are
            # allowed to read as prose.
            and node.value.count("/") == 1
            and " " in node.value
            and not node.value.startswith(("http", "/", "#", "."))
            and not node.value.endswith("/")
            and "\n" not in node.value and len(node.value) < 80
        ]
        assert not bad, f"{relative}: identifier literals contain prose: {bad}"


class TestRegionCoordinateSpace:
    """The page's size must not decide how its coordinates are read.

    The reader emits a 0-1000 grid whatever the page measures. Gating that on the page
    being larger than 1000px meant a smaller one had its grid values read as pixels,
    which neither raises nor drops a region -- it just puts every one of them in the
    wrong place, and only on small pages, so full-page benchmarks never saw it.
    """

    class _Img:
        def __init__(self, w, h):
            self.width, self.height = w, h

    @staticmethod
    def _centre(bounds):
        b = bounds.as_dict()
        return ((b["top_left"]["x"] + b["bottom_right"]["x"]) / 2,
                (b["top_left"]["y"] + b["bottom_right"]["y"]) / 2)

    @pytest.mark.parametrize("size", [(720, 960), (900, 700), (1190, 1540), (400, 400)])
    def test_grid_is_read_against_the_page(self, size):
        from interfaze_lite.contracts import parse_vlm_regions
        w, h = size
        region = parse_vlm_regions(
            '<p data-bbox="400 400 600 600">centre</p>', self._Img(w, h))[0]
        cx, cy = self._centre(region.bounds)
        assert abs(cx - w / 2) < 1 and abs(cy - h / 2) < 1

    def test_values_above_the_grid_are_still_pixels(self):
        from interfaze_lite.contracts import parse_vlm_regions
        region = parse_vlm_regions(
            '<p data-bbox="100 100 1400 1400">px</p>', self._Img(1600, 1600))[0]
        assert region.bounds.as_dict()["top_left"] == {"x": 100, "y": 100}

    def test_fractional_values_are_read_as_zero_to_one(self):
        from interfaze_lite.contracts import parse_vlm_regions
        region = parse_vlm_regions(
            '<p data-bbox="0.4 0.4 0.6 0.6">centre</p>', self._Img(800, 600))[0]
        cx, cy = self._centre(region.bounds)
        assert abs(cx - 400) < 1 and abs(cy - 300) < 1


class TestEmptyTaskTag:
    """An empty tag routes nothing, which is what omitting it also does.

    Templates that always emit `<task>{{ task }}</task>` produce `<task></task>` when
    the slot is unset. Treating that as an invalid task turned a valid request into a
    400 over an empty string.
    """

    def test_empty_tag_is_no_task(self):
        from interfaze_lite.validate import extract_task
        assert extract_task([{"role": "system", "content": "<task></task>"}]) is None

    def test_whitespace_only_tag_is_no_task(self):
        from interfaze_lite.validate import extract_task
        assert extract_task([{"role": "system", "content": "<task>   </task>"}]) is None

    def test_a_named_task_still_routes(self):
        from interfaze_lite.validate import extract_task
        assert extract_task([{"role": "system", "content": "<task>ocr</task>"}]) == "ocr"

    def test_an_unknown_name_is_still_rejected(self):
        from interfaze_lite.validate import RequestError, extract_task
        with pytest.raises(RequestError):
            extract_task([{"role": "system", "content": "<task>web_search</task>"}])


class TestEvalsReadTheKeysWeEmit:
    """Benchmarks must read the field names the tools actually return.

    Renaming a precontext key updated the service and its unit tests but not the
    eval scripts, which went on reading the old name and reported zero detections
    for every case -- a clean, plausible table of wrong numbers. Nothing failed;
    the benchmark simply measured a key that no longer existed.
    """

    def test_detection_evals_use_the_emitted_key(self):
        import pathlib
        emitted = pathlib.Path(
            "src/interfaze_lite/tools/detection.py").read_text()
        assert '"detected_objects": objects' in emitted, (
            "object_detection no longer emits detected_objects; update the evals")

        stale = []
        for script in pathlib.Path("evals").glob("*.py"):
            text = script.read_text()
            for line_no, line in enumerate(text.splitlines(), 1):
                # A lone legacy read is stale. One paired with the current name is a
                # deliberate fallback for older deployments.
                if 'get("objects")' in line and "detected_objects" not in text[
                        max(0, text.find(line) - 200):text.find(line) + 200]:
                    stale.append(f"{script.name}:{line_no}")
        assert not stale, f"evals read a key the tool no longer emits: {stale}"


class TestRoleContractMatchesInterfaze:
    """The accepted roles, and the alias, are the same four interfaze takes.

    `developer` is what the OpenAI SDKs send in place of `system` for recent models,
    so a caller that upgraded its SDK started getting "unsupported message role" from
    an otherwise identical request.
    """

    def test_developer_is_accepted(self):
        from interfaze_lite.validate import validate_messages
        validate_messages([{"role": "developer", "content": "be terse"},
                           {"role": "user", "content": "hi"}])

    def test_developer_becomes_system(self):
        from interfaze_lite.validate import normalise_roles
        out = normalise_roles([{"role": "developer", "content": "be terse"},
                               {"role": "user", "content": "hi"}])
        assert [m["role"] for m in out] == ["system", "user"]

    def test_a_task_tag_in_a_developer_turn_still_routes(self):
        """The reason normalising has to happen before anything reads a role."""
        from interfaze_lite.validate import extract_task, normalise_roles
        messages = [{"role": "developer", "content": "<task>ocr</task>"},
                    {"role": "user", "content": "read it"}]
        assert extract_task(messages) is None, "precondition: raw developer is not seen"
        assert extract_task(normalise_roles(messages)) == "ocr"

    def test_other_roles_are_untouched(self):
        from interfaze_lite.validate import normalise_roles
        original = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
        assert normalise_roles(original) == original

    def test_an_unknown_role_is_still_rejected(self):
        from interfaze_lite.validate import RequestError, validate_messages
        with pytest.raises(RequestError):
            validate_messages([{"role": "wizard", "content": "hi"}])


class TestDetectionRunsInWorkerProcesses:
    """Every detector call goes through the worker-process runtime.

    As threads, the engines shared Paddle's process-wide oneDNN context; a predictor
    read its cache unlocked while another wrote it, and under concurrent OCR the whole
    perception process segfaulted (MkldnnPostReset -> GetCachedObjectsNumber).
    """

    @staticmethod
    def _body(func_name: str) -> str:
        import pathlib
        tree = ast.parse(pathlib.Path(
            "src/interfaze_lite/modeling_interfaze_lite.py").read_text())
        func = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func_name)
        return ast.unparse(func)

    @pytest.mark.parametrize("func,call", [
        ("_detect_layout", "layout_pages"),
        ("_detect_text_lines", "line_rows"),
    ])
    def test_calls_go_to_the_worker_processes(self, func, call):
        assert f"self._detection_runtime().run({call}" in self._body(func)

    def test_no_engine_is_built_in_this_process(self):
        import pathlib

        source = pathlib.Path("src/interfaze_lite/modeling_interfaze_lite.py").read_text()
        assert "PaddleOCR(" not in source and "LayoutDetection(" not in source

class TestModelInitIsComplete:
    """`__init__` must still assign everything the rest of the class reads.

    An edit that inserted methods into the middle of `__init__` left every later
    assignment orphaned inside the last one. The module imported, the tests passed
    and OCR worked -- only transcription failed, with
    `'InterfazeLiteModel' object has no attribute '_dtype'`, because that was the
    first path to read an attribute the constructor no longer set.
    """

    @staticmethod
    def _init_assignments() -> set[str]:
        import pathlib
        tree = ast.parse(pathlib.Path(
            "src/interfaze_lite/modeling_interfaze_lite.py").read_text())
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "InterfazeLiteModel")
        init = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        names = {t.attr for node in ast.walk(init) if isinstance(node, ast.Assign)
                 for t in node.targets if isinstance(t, ast.Attribute)}
        # `self._cache: dict[...] = {}` is an AnnAssign, not an Assign.
        names |= {node.target.attr for node in ast.walk(init)
                  if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Attribute)}
        return names

    @pytest.mark.parametrize("attribute", [
        "_cache", "_cache_lock", "_dtype", "_torch_device",
        "_runtime", "_runtime_lock",
    ])
    def test_constructor_sets(self, attribute):
        assert attribute in self._init_assignments()

    def test_no_function_is_nested_inside_init(self):
        """A method defined inside `__init__` is what caused the truncation."""
        import pathlib
        tree = ast.parse(pathlib.Path(
            "src/interfaze_lite/modeling_interfaze_lite.py").read_text())
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == "InterfazeLiteModel")
        init = next(n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        nested = [n.name for n in init.body if isinstance(n, ast.FunctionDef)]
        assert not nested, f"methods defined inside __init__: {nested}"


class TestRecogniserOutput:
    def test_language_names_become_iso_codes(self):
        pytest.importorskip("transformers")
        from interfaze_lite.modeling_interfaze_lite import _language_code
        assert _language_code("english") == "en"
        assert _language_code("en") == "en"
        assert _language_code(None) is None

    def test_words_group_into_sentence_segments(self):
        pytest.importorskip("torch")
        from interfaze_lite.contracts import Word
        from interfaze_lite.modeling_interfaze_lite import _segments
        words = [Word("Hi,", 0.0, 0.3), Word("there.", 0.3, 0.6), Word("Next", 2.0, 2.3), Word("one", 2.3, 2.6)]
        spans = _segments(words)
        assert [(s.text, s.start, s.end) for s in spans] == [("Hi, there.", 0.0, 0.6), ("Next one", 2.0, 2.6)]


class TestReaderRegions:
    def test_label_and_markup_are_kept(self):
        from PIL import Image

        from interfaze_lite.contracts import parse_vlm_regions

        html = ('<div data-bbox="100 100 900 200" data-label="Table"><table><tr><td>a</td></tr></table></div>'
                '<div data-bbox="100 300 900 400" data-label="Text"><p>hello world</p></div>')
        regions = parse_vlm_regions(html, Image.new("RGB", (1000, 1000)))
        assert [r.category for r in regions] == ["Table", "Text"]
        assert regions[0].markup.startswith("<table>")
        assert regions[1].text == "hello world"


class TestSpeakerAlwaysNamed:
    def test_a_word_between_turns_takes_the_nearest_turn(self):
        from interfaze_lite.contracts import SpeakerTurn, Word, attribute_speakers
        turns = [SpeakerTurn("SPEAKER_00", 0.0, 10.0), SpeakerTurn("SPEAKER_01", 12.5, 13.0)]
        # Ends 0.5 s after the long turn, 2 s before the short one whose centre is nearer
        # by midpoint-to-midpoint distance than the long turn's.
        out = attribute_speakers([Word("gap", 10.2, 10.8)], turns, fill_nearest=True)
        assert out[0].speaker == "SPEAKER_00"

    def test_no_turns_at_all_is_still_a_named_speaker(self):
        from interfaze_lite.contracts import Word, attribute_speakers
        assert attribute_speakers([Word("hi", 0, 1)], [], fill_nearest=True)[0].speaker == "SPEAKER_00"


class TestWordSplit:
    def _bounds(self, tl, tr, br, bl):
        from interfaze_lite.contracts import Bounds, Point
        return Bounds(top_left=Point(*tl), top_right=Point(*tr), bottom_right=Point(*br),
                      bottom_left=Point(*bl), width=tr[0] - tl[0], height=bl[1] - tl[1])

    def test_words_join_back_into_the_line(self):
        from interfaze_lite.contracts import split_words
        words = split_words("See back of receipt", self._bounds((0, 0), (190, 0), (190, 20), (0, 20)), 0.9)
        assert " ".join(w.text for w in words) == "See back of receipt"
        assert [w.bounds.top_left.x for w in words] == [0, 40, 90, 120]
        assert words[-1].bounds.top_right.x == 190
        assert all(set(w.as_dict()) == {"text", "bounds", "confidence"} for w in words)

    def test_a_tilted_line_keeps_tilted_words(self):
        from interfaze_lite.contracts import split_words
        words = split_words("ab cd", self._bounds((0, 0), (50, 10), (50, 30), (0, 20)), 0.9)
        assert words[1].bounds.top_right.y == 10 and words[0].bounds.top_left.y == 0

    def test_one_word_keeps_the_line_box(self):
        from interfaze_lite.contracts import split_words
        b = self._bounds((0, 0), (40, 0), (40, 10), (0, 10))
        assert split_words("Total", b, 0.8)[0].bounds == b


class TestDuplicateBoxes:
    def _el(self, label, x1, y1, x2, y2):
        from interfaze_lite.contracts import Bounds, Point
        return {"label": label, "bounds": Bounds(top_left=Point(x1, y1), top_right=Point(x2, y1),
                bottom_right=Point(x2, y2), bottom_left=Point(x1, y2), width=x2 - x1, height=y2 - y1).as_dict()}

    def test_a_looped_box_collapses_to_the_first(self):
        from interfaze_lite.grounding import suppress_duplicates
        loop = [self._el("sword", 508, 409, 933, 458)] + [self._el("sword", 508, 410 + i % 3, 933, 457) for i in range(43)]
        assert len(suppress_duplicates(loop)) == 1

    def test_distinct_objects_and_labels_survive(self):
        from interfaze_lite.grounding import suppress_duplicates
        els = [self._el("dog", 0, 0, 100, 100), self._el("dog", 300, 0, 400, 100), self._el("collar", 0, 0, 100, 100)]
        assert len(suppress_duplicates(els)) == 3

    def test_a_ladder_of_abutting_identical_boxes_is_cut_to_its_first(self):
        """The same box stepped down the page by its own height: a loop, not a list."""
        from interfaze_lite.grounding import cut_ladders
        before = [self._el("icon", 10, 10, 50, 30)]
        ladder = [self._el("icon", 513, 150 + 13 * i, 553, 163 + 13 * i) for i in range(20)]
        kept = cut_ladders(before + ladder)
        assert kept == before + ladder[:1]

    def test_a_ladder_whose_boxes_jitter_in_size_is_cut_too(self):
        """The screenshot demo's dock: widths 55, 55, 52, ... each abutting the last."""
        from interfaze_lite.grounding import cut_ladders
        widths = [55, 55, 52, 55, 54, 53] * 3
        xs = [sum(widths[:i]) for i in range(len(widths) + 1)]
        ladder = [self._el("buttons", 20 + xs[i], 900, 20 + xs[i + 1], 960) for i in range(len(widths))]
        assert cut_ladders(ladder) == ladder[:1]

    def test_a_ladder_that_climbs_as_it_steps_is_cut(self):
        """A street photo's "person" boxes: 11 to 13 units wide, each starting where the
        last ended and 2 to 5 units higher, 40 of them to the edge of the image."""
        from interfaze_lite.grounding import cut_ladders
        person = self._el("person", 491, 487, 516, 586)
        widths, rises = [13, 12, 11, 13, 12, 13] * 4, [5, 2, 3, 2, 5, 2] * 4
        x, y, ladder = 531, 454, []
        for w, rise in zip(widths, rises):
            ladder.append(self._el("person", x, y, x + w, y + 68))
            x, y = x + w, y - rise
        assert cut_ladders([person] + ladder) == [person, ladder[0]]

    def test_a_ladder_running_leftwards_is_cut(self):
        """Each box ending where the last began: 20 boxes 15 to 18 units wide."""
        from interfaze_lite.grounding import cut_ladders
        widths = [17, 18, 17, 15, 18] * 4
        x, ladder = 983, []
        for w in widths:
            ladder.append(self._el("person", x - w, 654, x, 765))
            x -= w
        assert cut_ladders(ladder) == ladder[:1]

    def test_abutting_boxes_of_different_sizes_survive(self):
        """A tab bar: side by side, but each its own width."""
        from interfaze_lite.grounding import cut_ladders
        widths = [60, 90, 45, 120, 70, 100, 50]
        xs = [sum(widths[:i]) for i in range(len(widths) + 1)]
        tabs = [self._el("tab", 10 + xs[i], 50, 10 + xs[i + 1], 80) for i in range(len(widths))]
        assert cut_ladders(tabs) == tabs

    def test_a_real_list_with_gaps_and_a_short_stack_survive(self):
        from interfaze_lite.grounding import cut_ladders
        thumbnails = [self._el("icon", 63, 143 + 28 * i, 93, 169 + 28 * i) for i in range(7)]
        short_stack = [self._el("row", 300, 100 + 20 * i, 600, 120 + 20 * i) for i in range(5)]
        assert cut_ladders(thumbnails + short_stack) == thumbnails + short_stack


class TestMarkdownWithoutFigureLinks:
    def test_a_figure_keeps_its_description_and_loses_its_dead_link(self, monkeypatch):
        """Upstream links each figure to a file it saves itself; this service returns none."""
        import sys
        import types

        def parse_markdown(html, include_headers_footers=False, include_images=True):
            return ("See back of receipt\n\n![Walmart logo](7e8f_2_img.webp)\n\n"
                    "![](7e8f_5_img.webp)\n\n317-851-1102")

        monkeypatch.setitem(sys.modules, "chandra", types.ModuleType("chandra"))
        monkeypatch.setitem(sys.modules, "chandra.output",
                            types.SimpleNamespace(parse_markdown=parse_markdown))
        from interfaze_lite.contracts import vlm_markdown

        assert vlm_markdown("<div/>") == (
            "See back of receipt\n\n[Image: Walmart logo]\n\n\n\n317-851-1102")


class TestSpeechWindows:
    RATE = 16_000

    def _speech(self, seconds):
        import numpy as np
        t = np.arange(int(seconds * self.RATE)) / self.RATE
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_windows_are_cut_in_the_quiet_and_never_exceed_30_s(self):
        import numpy as np
        from interfaze_lite.contracts import speech_windows

        # Speech in 24 s stretches with a second of silence between them.
        gap = np.zeros(self.RATE, dtype=np.float32)
        audio = np.concatenate([self._speech(24), gap, self._speech(24), gap, self._speech(24)])
        windows = speech_windows(audio, self.RATE)
        assert all(e - s <= 30 * self.RATE for s, e in windows)
        assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
        # Every sample of speech is in a window: no cut fell inside a stretch of it.
        covered = np.zeros(len(audio), dtype=bool)
        for s, e in windows:
            covered[s:e] = True
        assert covered[np.abs(audio) > 0].all()

    def test_silence_around_speech_is_trimmed(self):
        import numpy as np
        from interfaze_lite.contracts import speech_windows

        audio = np.concatenate([np.zeros(3 * self.RATE, dtype=np.float32), self._speech(10),
                                np.zeros(3 * self.RATE, dtype=np.float32)])
        [(s, e)] = speech_windows(audio, self.RATE)
        assert 2.7 * self.RATE <= s <= 3 * self.RATE and 13 * self.RATE <= e <= 13.3 * self.RATE

    def test_a_silent_window_is_left_out(self):
        import numpy as np
        from interfaze_lite.contracts import speech_windows

        audio = np.concatenate([self._speech(25), np.zeros(40 * self.RATE, dtype=np.float32),
                                self._speech(25)])
        windows = speech_windows(audio, self.RATE)
        assert len(windows) >= 2
        for s, e in windows:
            assert np.abs(audio[s:e]).max() > 0

    def test_short_audio_is_one_window(self):
        from interfaze_lite.contracts import speech_windows

        audio = self._speech(12)
        [(s, e)] = speech_windows(audio, self.RATE)
        assert s == 0 and len(audio) - e < 0.05 * self.RATE

    def test_a_batch_that_does_not_fit_is_freed_before_its_halves_run(self):
        """Halved inside the except clause, the failed batch's tensors stayed reachable
        through the live exception, and every half ran out of memory down to one window."""
        import weakref

        import numpy as np
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from interfaze_lite.contracts import speech_windows
        from interfaze_lite.modeling_interfaze_lite import _transcribe_windows

        class Tensors:
            pass

        failed, alive_at_call, sizes = [], [], []

        def asr(inputs, **kwargs):
            alive_at_call.append(any(ref() is not None for ref in failed))
            sizes.append(len(inputs))
            if len(inputs) > 2:
                held = Tensors()  # what the failed batch had allocated, live in its frame
                failed.append(weakref.ref(held))
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return [{"chunks": [{"text": "hi", "timestamp": (0.0, 1.0)}]} for _ in inputs]

        gap = np.zeros(self.RATE, dtype=np.float32)
        audio = np.concatenate([np.concatenate([self._speech(24), gap]) for _ in range(8)])
        units, _ = _transcribe_windows(asr, audio, "en", words=True, batch_size=8)
        assert failed, "a batch ran out of memory"
        assert not any(alive_at_call)
        assert len(units) == len(speech_windows(audio, self.RATE))
        # Once two fit, the rest of the recording goes two at a time.
        assert sizes[:2] == [8, 4] and all(n <= 2 for n in sizes[2:])


class TestPhotoWithoutText:
    """Asked for the text of a giraffe photo, the reader's description came back as its text."""

    def test_a_picture_description_with_no_lines_read_is_no_text(self):
        from interfaze_lite.contracts import describes_a_photo
        assert describes_a_photo([], "[Image: A photograph of a giraffe and its calf.]A photograph of a giraffe")

    def test_a_page_the_detector_read_keeps_its_text(self):
        from interfaze_lite.contracts import describes_a_photo
        assert not describes_a_photo(["Cocoa Powder"], "[Image: Ingredients for a dessert.]")

    def test_text_the_detector_missed_is_kept(self):
        from interfaze_lite.contracts import describes_a_photo
        assert not describes_a_photo([], "Dear Sam, thank you for the flowers.")


class TestEstimatedWords:
    """Speakers joined to a recording too long to decode word timing for."""

    def test_a_segment_is_shared_among_its_words_by_length(self):
        from interfaze_lite.contracts import Word, estimate_words

        words = estimate_words([Word("hi there everyone", 10.0, 13.0)])
        assert [w.text for w in words] == ["hi", "there", "everyone"]
        assert words[0].start == 10.0 and words[-1].end == pytest.approx(13.0)
        assert all(a.end == pytest.approx(b.start) for a, b in zip(words, words[1:]))
        assert words[2].end - words[2].start > words[0].end - words[0].start

    def test_a_change_of_speaker_inside_a_segment_splits_it(self):
        from interfaze_lite.contracts import (SpeakerTurn, Word, attribute_speakers, estimate_words,
                                              group_by_speaker)

        segment = Word("yes I agree completely and what about Friday then", 0.0, 6.0)
        turns = [SpeakerTurn("A", 0.0, 3.5), SpeakerTurn("B", 3.5, 6.0)]
        chunks = group_by_speaker(attribute_speakers(estimate_words([segment]), turns))
        assert [c["speaker"] for c in chunks] == ["A", "B"]
        assert chunks[1]["text"].endswith("Friday then")

    def test_text_without_spaces_stays_one_unit(self):
        from interfaze_lite.contracts import Word, estimate_words

        assert [(w.text, w.start, w.end) for w in estimate_words([Word("你好世界", 1.0, 2.0)])] == [("你好世界", 1.0, 2.0)]


class TestTranscriptChunks:
    def test_segments_are_cut_to_three_seconds_as_interfaze_cuts_them(self):
        """interfaze's own chunks for the sample clip, boundaries and words exactly."""
        from interfaze_lite.contracts import split_long_chunks
        segments = [
            {"text": "The little tales they tell are false", "timestamp": [0.0, 4.78]},
            {"text": "The door was barred, locked and bolted as well", "timestamp": [4.78, 9.48]},
            {"text": "Ripe pears are fit for a queen's table", "timestamp": [9.48, 13.06]},
        ]
        assert split_long_chunks(segments) == [
            {"text": "The little tales", "timestamp": [0.0, 2.39]},
            {"text": "they tell are false", "timestamp": [2.39, 4.78]},
            {"text": "The door was barred,", "timestamp": [4.78, 7.13]},
            {"text": "locked and bolted as well", "timestamp": [7.13, 9.48]},
            {"text": "Ripe pears are fit", "timestamp": [9.48, 11.27]},
            {"text": "for a queen's table", "timestamp": [11.27, 13.06]},
        ]

    def test_short_segments_words_and_untimed_chunks_are_left_alone(self):
        from interfaze_lite.contracts import split_long_chunks
        chunks = [{"text": "Hello there", "timestamp": [0.0, 2.5]}, {"text": "word", "timestamp": [2.5, 6.0]},
                  {"text": "no end time yet", "timestamp": [6.0, None]}]
        assert split_long_chunks(chunks) == chunks
