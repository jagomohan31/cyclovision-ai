"""
Download a MOSDAC satellite-data order over SFTP.

This automates the WinSCP steps: connect to download.mosdac.gov.in, find your
order folder, and pull everything down -- but as a script instead of manual
clicking, so a 40GB / 2,818-file job doesn't need you babysitting it.

Two things this adds over dragging files in WinSCP:
  - --explore mode: prints the remote folder tree so you can actually find
    your Request ID's folder instead of guessing.
  - Safe resume: if it gets interrupted, just run the exact same command
    again. Files you already have (verified by size) are skipped, and the
    connection auto-reconnects if it drops mid-run.

SETUP (one time):
    pip install paramiko

STEP 1 - find your order folder:
    python download_mosdac_order.py --explore
    python download_mosdac_order.py --explore --remote "some/folder" --depth 3

STEP 2 - download it:
    python download_mosdac_order.py --remote "Sep2026_190662" --local "./data/raw/mosdac_order"

If it stops for any reason (connection drop, laptop sleeps, you close the
terminal), just run that exact same command again.
"""

import argparse
import getpass
import os
import stat
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    raise SystemExit(
        "Missing dependency. Run this first:\n    pip install paramiko"
    )

HOST = "download.mosdac.gov.in"
PORT = 22
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5


class MosdacConnection:
    """A reconnectable SFTP session. A transfer this size will very likely
    hit at least one dropped connection along the way -- this reconnects
    instead of the whole script dying."""

    def __init__(self, username, password):
        self.username = username 
        self.password = password
        self.client = None
        self.sftp = None
        self.reconnect()

    def reconnect(self):
        for conn in (self.sftp, self.client):
            try:
                conn.close()
            except Exception:
                pass
        self.client = paramiko.SSHClient()
        # Auto-accepts the host key, same as clicking "Yes" on WinSCP's
        # first-connection warning -- fine here since HOST is fixed above.
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            HOST,
            port=PORT,
            username=self.username,
            password=self.password,
            timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
        self.sftp = self.client.open_sftp()

    def close(self):
        try:
            self.sftp.close()
            self.client.close()
        except Exception:
            pass


def _join(remote_path, name):
    return name if remote_path in (".", "") else f"{remote_path.rstrip('/')}/{name}"


def explore(conn, path, depth, indent=0):
    """Print the remote folder tree so you can find your order folder."""
    try:
        entries = conn.sftp.listdir_attr(path)
    except IOError as e:
        print("  " * indent + f"[can't open {path}: {e}]")
        return
    for entry in sorted(entries, key=lambda e: e.filename):
        full_path = _join(path, entry.filename)
        is_dir = stat.S_ISDIR(entry.st_mode)
        size_note = "" if is_dir else f"  ({entry.st_size / 1e6:.1f} MB)"
        print("  " * indent + entry.filename + ("/" if is_dir else "") + size_note)
        if is_dir and depth > 0:
            explore(conn, full_path, depth - 1, indent + 1)


def list_remote_files(conn, remote_path):
    """Recursively yield (path, size) for every file under remote_path."""
    for entry in conn.sftp.listdir_attr(remote_path):
        full_path = _join(remote_path, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            yield from list_remote_files(conn, full_path)
        else:
            yield full_path, entry.st_size


def download_one(conn, remote_path, local_path, expected_size):
    if local_path.exists() and local_path.stat().st_size == expected_size:
        return "skipped"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(local_path.name + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn.sftp.get(remote_path, str(tmp_path))
            tmp_path.replace(local_path)
            return "downloaded"
        except Exception as e:
            print(f"        attempt {attempt}/{MAX_RETRIES} failed ({e}); reconnecting...")
            time.sleep(RETRY_WAIT_SECONDS)
            try:
                conn.reconnect()
            except Exception as reconnect_err:
                print(f"        reconnect failed: {reconnect_err}")
    return "failed"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--explore", action="store_true",
                         help="Browse remote folders, then exit (no downloading)")
    parser.add_argument("--remote", default=".",
                         help="Remote folder to download or browse. Default: your SFTP home folder.")
    parser.add_argument("--local", default="./mosdac_download",
                         help="Local folder to save into")
    parser.add_argument("--depth", type=int, default=2,
                         help="How many folder levels deep --explore prints (default 2)")
    args = parser.parse_args()

    username = os.environ.get("MOSDAC_USERNAME") or input("MOSDAC username: ").strip()
    password = os.environ.get("MOSDAC_PASSWORD") or getpass.getpass("MOSDAC password: ")

    conn = MosdacConnection(username, password)

    try:
        if args.explore:
            print(f"\nBrowsing '{args.remote}' on {HOST} ({args.depth} levels deep):\n")
            explore(conn, args.remote, args.depth)
            print("\nOnce you spot your order folder, run:")
            print(f'  python {Path(__file__).name} --remote "path/you/found" '
                  f'--local "./data/raw/mosdac_order"')
            return

        print(f"Scanning '{args.remote}' on {HOST} -- this can take a minute for thousands of files...")
        files = list(list_remote_files(conn, args.remote))
        total_bytes = sum(size for _, size in files)
        print(f"Found {len(files)} files, {total_bytes / 1e9:.2f} GB total.\n")

        local_root = Path(args.local)
        counts = {"downloaded": 0, "skipped": 0, "failed": 0}
        done_bytes = 0
        prefix = "" if args.remote in (".", "") else args.remote.rstrip("/") + "/"

        for i, (remote_file, size) in enumerate(files, 1):
            rel_path = remote_file[len(prefix):] if remote_file.startswith(prefix) else remote_file
            local_file = local_root / rel_path

            result = download_one(conn, remote_file, local_file, size)
            counts[result] += 1
            done_bytes += size

            print(f"[{i}/{len(files)}] {result:>10s}  {rel_path}  "
                  f"({done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB)")

        print(f"\nDone: {counts['downloaded']} downloaded, {counts['skipped']} already had, "
              f"{counts['failed']} failed.")
        if counts["failed"]:
            print("Re-run the exact same command -- completed files are skipped "
                  "automatically, so it'll only retry what actually failed.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
