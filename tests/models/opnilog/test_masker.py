# Local
from opni_inference_service.models.opnilog import masker


def test_masker():
    # TODO: 5.5e-3 should be masked as <NUM> instead of <TOKEN_WITH_DIGITS>
    log_masker = masker.LogMasker()
    content = [
        "http://www.example.com/somewhere/over/the/rainbow.go : 12345\n",
        "some@example.com | 127.0.0.1/24 | 12.34 s",
        "/bin/bash | 6500 | abc123",
    ]
    expected_output = [
        "<url> : <num>",
        "<email_address> <ip> <duration>",
        "<path> <num> <token_with_digit>",
    ]

    for i, c in enumerate(content):
        output = log_masker.mask(c)
        assert output == expected_output[i]
