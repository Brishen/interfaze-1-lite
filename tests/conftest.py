import pytest

from interfaze_lite.tools import ocr


@pytest.fixture(autouse=True)
def _no_reads_kept_between_tests():
    """Each test mocks its own backend; a document another test read is not this one's."""
    ocr._reads.clear()
    yield
    ocr._reads.clear()
