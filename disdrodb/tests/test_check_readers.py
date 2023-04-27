from disdrodb.l0.check_readers import check_all_readers


def test_check_all_readers(tmpdir):
    """Test test_all_readers."""

    check_all_readers()
