import socket

from kakeya.lab import (
    _public_url,
    _same_local_host,
    _version_at_least,
    build_parser,
    port_available,
)


def test_launcher_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.api_port == 8000
    assert args.ui_port == 3000
    assert args.install is False
    assert args.no_browser is False


def test_version_comparison() -> None:
    assert _version_at_least("22.13.0", (22, 13, 0))
    assert _version_at_least("23.0.1", (22, 13, 0))
    assert not _version_at_least("22.12.9", (22, 13, 0))
    assert not _version_at_least(None, (22, 13, 0))


def test_public_urls_and_local_host_equivalence() -> None:
    assert _public_url("0.0.0.0", 3000) == "http://127.0.0.1:3000"
    assert _same_local_host("localhost", "127.0.0.1")
    assert not _same_local_host("192.168.1.10", "127.0.0.1")


def test_port_availability_detects_bound_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]

        assert not port_available("127.0.0.1", port)

    assert port_available("127.0.0.1", port)
