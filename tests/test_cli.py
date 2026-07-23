from train import build_argument_parser


def test_parser_has_beginner_friendly_defaults():
    parser = build_argument_parser()
    arguments = parser.parse_args([])

    assert arguments.batch_size == 128
    assert arguments.epochs == 5
    assert arguments.learning_rate == 0.30


def test_parser_reads_custom_values():
    parser = build_argument_parser()
    arguments = parser.parse_args(
        ["--epochs", "2", "--train-size", "100", "--test-size", "50"]
    )

    assert arguments.epochs == 2
    assert arguments.train_size == 100
    assert arguments.test_size == 50
