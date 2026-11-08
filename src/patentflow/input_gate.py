"""Verify that this user's input files are not currently open for writing."""
import os
from pathlib import Path


def writable_handles(paths):
    targets = {str(Path(p).resolve()) for p in paths}
    writers = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            for fd in (process / 'fd').iterdir():
                try:
                    link = os.readlink(str(fd))
                    if link not in targets:
                        continue
                    info = (process / 'fdinfo' / fd.name).read_text()
                    flags = next(line.split()[1] for line in info.splitlines() if line.startswith('flags:'))
                    if int(flags, 8) & os.O_ACCMODE in (os.O_WRONLY, os.O_RDWR):
                        writers.append({'pid': int(process.name), 'path': link})
                except (OSError, StopIteration):
                    continue
        except OSError:
            continue
    return writers


def snapshot(paths):
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            if p.exists() else None for p in paths}
