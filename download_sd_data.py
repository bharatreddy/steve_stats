#!/usr/bin/env python3
"""Download SuperDARN fitacf3 and map2 data for dates in event_list.py.

Authentication is handled by OpenSSH's ``sftp`` command.  If public-key
authentication is not configured, OpenSSH securely prompts for the account
password in the terminal before logging in.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

try:
    import pexpect
except ImportError:  # Report a useful error from main instead of at import time.
    pexpect = None

from event_list import selected_events


REMOTE = "sd-data@sd-data.ece.vt.edu"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
RADAR_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
# The execution environment's system-wide SSH config may be mounted with an
# owner that OpenSSH refuses to trust.  An empty explicit config bypasses that
# broken file while OpenSSH still uses its secure defaults and known_hosts.
SFTP_COMMAND = [
    "sftp",
    "-F",
    "/dev/null",
    "-o",
    "PreferredAuthentications=password,keyboard-interactive",
    "-o",
    "PubkeyAuthentication=no",
    REMOTE,
]


def parse_args() -> argparse.Namespace:
    events = selected_events()
    parser = argparse.ArgumentParser(
        description=(
            "Download full-day fitacf3 files from all radars and the northern "
            "map2 file for dates defined in event_list.py."
        )
    )
    parser.add_argument(
        "--event",
        dest="event_names",
        action="append",
        choices=sorted(events),
        help=(
            "download only this event (may be supplied more than once); "
            "the default is every event"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show which remote directories will be inspected without connecting",
    )
    return parser.parse_args()


def event_dates(event_names: Iterable[str] | None) -> list[dt.date]:
    events = selected_events()
    names = list(event_names) if event_names else list(events)
    return sorted({events[name].event_date.date() for name in names})


def sftp_quote(value: str) -> str:
    """Quote one argument for OpenSSH sftp's command parser."""
    if "\n" in value or "\r" in value:
        raise ValueError("SFTP paths cannot contain newlines")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def print_dry_run(dates: Iterable[dt.date], output_dir: Path) -> None:
    for event_date in dates:
        year = event_date.strftime("%Y")
        day = event_date.strftime("%Y%m%d")
        date_dir = (output_dir / day).resolve()
        print(
            f"List /sd-data/{year}/fitacf3, inspect every radar for "
            f"{day}.*.fitacf3.bz2, and download exact filenames to {date_dir}"
        )
        if not (date_dir / f"{day}.north.map2.bz2").is_file():
            print(
                f"Download /sd-data/{year}/map2/north/"
                f"{day}.north.map2.bz2 to {date_dir}"
            )


def radar_from_filename(filename: str, day: str) -> str | None:
    """Return the radar field from a standard fitacf3 filename."""
    # Both forms exist in the archive:
    # yyyymmdd.hhmm.VALUE.RADAR.fitacf3.bz2
    # yyyymmdd.hhmm.VALUE.RADAR.CHANNEL.fitacf3.bz2
    parts = filename.split(".")
    if (
        len(parts) in (6, 7)
        and parts[0] == day
        and parts[-2:] == ["fitacf3", "bz2"]
        and RADAR_NAME.fullmatch(parts[3])
    ):
        return parts[3]
    return None


def migrate_legacy_layout(
    dates: Iterable[dt.date], output_dir: Path
) -> tuple[int, int, list[Path]]:
    """Move files produced by the first script version into the new layout."""
    map2_moved = 0
    fitacf_moved = 0
    conflicts: list[Path] = []

    for event_date in dates:
        day = event_date.strftime("%Y%m%d")
        date_dir = output_dir / day

        old_map = output_dir / "map2" / day / f"{day}.north.map2.bz2"
        if old_map.is_file():
            date_dir.mkdir(parents=True, exist_ok=True)
            new_map = date_dir / old_map.name
            if new_map.exists():
                conflicts.append(old_map)
            else:
                old_map.replace(new_map)
                map2_moved += 1

        old_fitacf_dir = output_dir / "fitacf3" / day
        if old_fitacf_dir.is_dir():
            for source in old_fitacf_dir.glob(f"{day}.*.fitacf3.bz2"):
                radar = radar_from_filename(source.name, day)
                if radar is None:
                    conflicts.append(source)
                    continue
                radar_dir = date_dir / radar
                radar_dir.mkdir(parents=True, exist_ok=True)
                target = radar_dir / source.name
                if target.exists():
                    conflicts.append(source)
                else:
                    source.replace(target)
                    fitacf_moved += 1

        for old_dir in (old_fitacf_dir, old_map.parent):
            try:
                old_dir.rmdir()
            except OSError:
                pass

    for old_root in (output_dir / "fitacf3", output_dir / "map2"):
        try:
            old_root.rmdir()
        except OSError:
            pass

    return map2_moved, fitacf_moved, conflicts


class SFTPError(RuntimeError):
    """Raised when the interactive SFTP session cannot complete a command."""


def connect_sftp():
    """Start one SFTP session and authenticate without exposing the password."""
    child = pexpect.spawn(
        SFTP_COMMAND[0],
        SFTP_COMMAND[1:],
        encoding="utf-8",
        timeout=30,
        echo=False,
    )
    password_attempts = 0

    while True:
        result = child.expect(
            [
                r"(?i)are you sure you want to continue connecting.*\?",
                r"(?i)password:",
                r"sftp> ",
                r"(?i)permission denied",
                pexpect.EOF,
                pexpect.TIMEOUT,
            ]
        )
        if result == 0:
            print(child.before + child.after, end="", flush=True)
            answer = input("\nType yes to trust this host key: ").strip()
            child.sendline(answer)
        elif result == 1:
            password_attempts += 1
            if password_attempts > 3:
                child.close(force=True)
                raise SFTPError("Authentication failed after three password attempts.")
            password = getpass.getpass(f"Password for {REMOTE}: ")
            child.sendline(password)
            del password
        elif result == 2:
            return child
        elif result == 3:
            child.close(force=True)
            raise SFTPError("The server rejected the username or password.")
        elif result == 4:
            details = child.before.strip()
            child.close(force=True)
            raise SFTPError(f"SFTP closed during login: {details or 'no details'}")
        else:
            child.close(force=True)
            raise SFTPError("Timed out while connecting to the SFTP server.")


def run_sftp_command(child, command: str, timeout: int = 3600) -> str:
    """Run one command and return its terminal output without the prompt."""
    child.sendline(command)
    result = child.expect([r"sftp> ", pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
    output = child.before.replace("\r", "")
    if result == 0:
        return output
    if result == 1:
        raise SFTPError(f"SFTP disconnected while running: {command}\n{output.strip()}")
    raise SFTPError(f"SFTP timed out while running: {command}")


def listing_lines(output: str) -> list[str]:
    """Remove diagnostic lines from the output of an SFTP ls command."""
    ignored = (
        "Can't ",
        "Couldn't ",
        "File ",
        "Fetching ",
        "Connected ",
        "remote ",
        "stat ",
        "ls ",
        "cd ",
        "get ",
        "reget ",
        "sftp>",
    )
    return [
        line.strip()
        for line in output.splitlines()
        if line.strip()
        and not line.strip().startswith(ignored)
        and "not found" not in line.lower()
    ]


def list_radars(child, year: str) -> list[str]:
    remote_dir = f"/sd-data/{year}/fitacf3"
    cd_output = run_sftp_command(child, f"cd {remote_dir}")
    if "couldn't" in cd_output.lower() or "not found" in cd_output.lower():
        raise SFTPError(f"Could not open {remote_dir}.\n{cd_output.strip()}")

    # Listing the current directory avoids SFTP returning the directory path
    # itself instead of its contents. Split tokens as a fallback for servers
    # that format the listing in columns despite -1.
    output = run_sftp_command(child, "ls -1")
    radars = {
        token.rstrip("/").rsplit("/", 1)[-1]
        for line in listing_lines(output)
        for token in line.split()
        if RADAR_NAME.fullmatch(token.rstrip("/").rsplit("/", 1)[-1])
        and token.rstrip("/").rsplit("/", 1)[-1] != "fitacf3"
    }
    if not radars:
        raise SFTPError(f"No radar directories were found in {remote_dir}.\n{output.strip()}")
    return sorted(radars)


def list_fitacf_files(child, year: str, radar: str, day: str) -> list[str]:
    radar_dir = f"/sd-data/{year}/fitacf3/{radar}"
    # This unquoted wildcard is used only for listing within one known radar.
    output = run_sftp_command(
        child, f"ls -1 {radar_dir}/{day}.*.fitacf3.bz2"
    )
    matches: list[str] = []
    for line in listing_lines(output):
        if any(character in line for character in "*?["):
            continue
        remote_file = line if line.startswith("/") else f"{radar_dir}/{line}"
        filename = remote_file.rsplit("/", 1)[-1]
        if (
            filename.startswith(f"{day}.")
            and filename.endswith(".fitacf3.bz2")
            and " " not in filename
        ):
            matches.append(remote_file)
    return sorted(set(matches))


def download_exact_file(child, remote_file: str, local_file: Path) -> None:
    """Download or resume one exact remote filename and verify local output."""
    local_file.parent.mkdir(parents=True, exist_ok=True)
    operation = "reget" if local_file.exists() else "get"
    output = run_sftp_command(
        child,
        f"{operation} -p {sftp_quote(remote_file)} {sftp_quote(str(local_file))}",
    )
    if not local_file.is_file() or local_file.stat().st_size == 0:
        raise SFTPError(
            f"Download did not create {local_file}.\nRemote output: {output.strip()}"
        )


def download_all(child, dates: Iterable[dt.date], output_dir: Path) -> int:
    """Enumerate radar directories and download every matching fitacf3 file."""
    radar_cache: dict[str, list[str]] = {}
    downloaded = 0

    for event_date in dates:
        year = event_date.strftime("%Y")
        day = event_date.strftime("%Y%m%d")
        date_dir = output_dir / day
        date_dir.mkdir(parents=True, exist_ok=True)

        if year not in radar_cache:
            radar_cache[year] = list_radars(child, year)
            print(
                f"{year}: found {len(radar_cache[year])} radar directories: "
                + ", ".join(radar_cache[year])
            )

        date_matches = 0
        for radar in radar_cache[year]:
            remote_files = list_fitacf_files(child, year, radar, day)
            if not remote_files:
                continue
            print(f"{day} {radar}: downloading {len(remote_files)} file(s).")
            for remote_file in remote_files:
                local_file = date_dir / radar / remote_file.rsplit("/", 1)[-1]
                download_exact_file(child, remote_file, local_file)
                downloaded += 1
                date_matches += 1

        if date_matches == 0:
            print(f"Warning: no fitacf3 files found for {day}.", file=sys.stderr)

        map2_name = f"{day}.north.map2.bz2"
        map2_local = date_dir / map2_name
        if not map2_local.is_file():
            map2_remote = f"/sd-data/{year}/map2/north/{map2_name}"
            print(f"{day}: downloading map2 file.")
            download_exact_file(child, map2_remote, map2_local)

    return downloaded


def validate_downloads(dates: Iterable[dt.date], output_dir: Path) -> list[str]:
    missing: list[str] = []
    for event_date in dates:
        day = event_date.strftime("%Y%m%d")
        date_dir = output_dir / day
        fitacf_files = list(date_dir.glob(f"*/{day}.*.fitacf3.bz2"))
        map2_file = date_dir / f"{day}.north.map2.bz2"
        if not fitacf_files:
            missing.append(f"fitacf3 files for {day}")
        if not map2_file.is_file():
            missing.append(f"map2 file for {day}")
    return missing


def main() -> int:
    args = parse_args()
    dates = event_dates(args.event_names)

    print("Dates: " + ", ".join(day.isoformat() for day in dates))
    print(f"Destination: {args.output_dir.resolve()}")

    if args.dry_run:
        print_dry_run(dates, args.output_dir)
        return 0

    if shutil.which("sftp") is None:
        print(
            "Error: the OpenSSH sftp command is not installed or not on PATH.",
            file=sys.stderr,
        )
        return 2
    if pexpect is None:
        print(
            "Error: this script requires the Python 'pexpect' package.",
            file=sys.stderr,
        )
        return 2

    map2_moved, fitacf_moved, conflicts = migrate_legacy_layout(
        dates, args.output_dir
    )
    if map2_moved or fitacf_moved:
        print(
            f"Migrated {map2_moved} map2 and {fitacf_moved} fitacf3 file(s) "
            "from the previous folder layout."
        )
    if conflicts:
        print(
            "Could not migrate these existing files because their destination "
            "already exists or their radar is unknown:\n  - "
            + "\n  - ".join(str(path) for path in conflicts),
            file=sys.stderr,
        )
        return 1

    print(
        f"Connecting to {REMOTE}; enter the SSH password when prompted.",
        flush=True,
    )
    child = None
    try:
        child = connect_sftp()
        downloaded = download_all(child, dates, args.output_dir)
        child.sendline("quit")
        child.expect(pexpect.EOF, timeout=30)
    except SFTPError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDownload cancelled.", file=sys.stderr)
        return 130
    finally:
        if child is not None and child.isalive():
            child.close(force=True)

    missing = validate_downloads(dates, args.output_dir)
    if missing:
        print(
            "Download finished, but no local match was found for:\n  - "
            + "\n  - ".join(missing),
            file=sys.stderr,
        )
        return 1

    print(f"Download complete. Retrieved or resumed {downloaded} fitacf3 file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
