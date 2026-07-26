import os
import signal
import subprocess
import sys

import pytest

from kakeya.job_manager import _terminate_process_tree


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group behavior")
def test_terminate_process_tree_escalates_for_unresponsive_worker() -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import signal,time;"
                "signal.signal(signal.SIGTERM,lambda *_:None);"
                "print('ready',flush=True);"
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"

    forced = _terminate_process_tree(process, grace_seconds=0.1)
    process.wait(timeout=2)

    assert forced is True
    assert process.returncode == -signal.SIGKILL
