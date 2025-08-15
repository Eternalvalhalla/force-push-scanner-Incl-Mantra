from __future__ import annotations  # Postpone annotation evaluation for Python < 3.10 support

import sys
import sqlite3
import json
import tempfile
from datetime import timezone
import subprocess
import datetime as _dt
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional

# Stdlib additions
import argparse
import logging
from contextlib import suppress
import shutil
import re
import os
import csv

# Cross-platform color support (Windows, Linux, macOS)
try:
    from colorama import init as colorama_init, Fore, Style

    colorama_init()  # enables ANSI on Windows terminals
except ImportError:  # graceful degradation – no colors

    class _Dummy:
        def __getattr__(self, _):
            return ""

    Fore = Style = _Dummy()


def terminate(msg: str) -> None:
    """Exit the program with an error message (in red)."""
    print(f"{Fore.RED}[✗] {msg}{Style.RESET_ALL}")
    sys.exit(1)


class RunCmdError(RuntimeError):
    """Raised when an external command returns a non-zero exit status."""

def run(cmd: List[str], cwd: Path | None = None) -> str:
    """Execute *cmd* and return its *stdout* as *str*.

    If the command exits non-zero, a ``RunCmdError`` is raised so callers can
    decide whether to abort or ignore.
    """

    logging.debug("Running command: %s (cwd=%s)", " ".join(cmd), cwd or ".")
    try:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            env=env,
        )
        return proc.stdout
    except subprocess.CalledProcessError as err:
        raise RunCmdError(
            f"Command failed ({err.returncode}): {" ".join(cmd)}\n{err.stderr.strip()}"
        ) from err

def _compile_patterns(raw_patterns: List[str]) -> List[re.Pattern]:
    compiled_patterns = []
    for line in raw_patterns:
        pattern = line.strip()
        if pattern:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                print(f"[!] Invalid regex pattern (skipping): '{pattern}' - {e}")
    return compiled_patterns

# Utility: extract year from Unix epoch INT.
def to_year(date_val) -> str:  # type: ignore[override]
    """Return the four-digit year (YYYY) from *date_val* which can be an int (epoch)"""
    return _dt.datetime.fromtimestamp(int(date_val), tz=timezone.utc).strftime("%Y")

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")

############################################################
# Phase 1: Gather data from SQLite3 (default) or user-supplied CSV
############################################################

# Column names expected from SQLite3 / CSV export
_EXPECTED_FIELDS = {"repo_org","repo_name", "before", "timestamp"}


def _validate_row(input_org: str, row: dict, idx: int) -> tuple[str, str, int | str]:
    """Validate that *row* contains the required columns and return the tuple.

    Raises ``ValueError`` on validation failure so callers can abort early.
    """

    missing = _EXPECTED_FIELDS - row.keys()
    if missing:
        raise ValueError(f"Row {idx} is missing fields: {', '.join(sorted(missing))}")

    repo_org = str(row["repo_org"]).strip()
    repo_name = str(row["repo_name"]).strip()
    before = str(row["before"]).strip()
    ts = row["timestamp"]

    if not repo_org:
        raise ValueError(f"Row {idx} – 'repo_org' is empty")
    if repo_org != input_org:
        raise ValueError(f"Row {idx} – 'repo_org' does not match 'input_org': {repo_org} != {input_org}")
    if not repo_name:
        raise ValueError(f"Row {idx} – 'repo_name' is empty")
    if not _SHA_RE.fullmatch(before):
        raise ValueError(f"Row {idx} – 'before' does not look like a commit SHA")

    # BigQuery exports numeric INT64 as str when using CSV, accommodate both.
    try:
        ts_int: int | str = int(ts)
    except Exception as exc:
        raise ValueError(f"Row {idx} – 'timestamp' must be int, got {ts!r}") from exc

    return repo_org, repo_name, before, ts_int


def _gather_from_iter(input_org: str, rows: List[dict]) -> Dict[str, List[dict]]:
    """Convert iterable rows into the internal repos mapping."""
    repos: Dict[str, List[dict]] = defaultdict(list)
    for idx, row in enumerate(rows, 1):
        try:
            repo_org, repo_name, before, ts_int = _validate_row(input_org, row, idx)
        except ValueError as ve:
            terminate(str(ve))

        url = f"https://github.com/{repo_org}/{repo_name}"
        repos[url].append({"before": before, "date": ts_int})
    if not repos:
        terminate("No force-push events found for that user – dataset empty")
    return repos


def gather_commits(
    input_org: str,
    events_file: Optional[Path] | None = None,
    db_file: Optional[Path] | None = None,
) -> Dict[str, List[dict]]:
    """Return mapping of repo URL → list[{before, pushed_at}].

    The data can be sourced either from:
    1. A CSV export (``--events-file``)
    2. The pre-built SQLite database downloaded via the Google Form (``--db-file``)

    Both sources expose the columns: repo_org, repo_name, before, timestamp.
    """

    if events_file is not None:
        if not events_file.exists():
            terminate(f"Events file not found: {events_file}")
        rows: List[dict] = []
        try:
            with events_file.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception as exc:
            terminate(f"Failed to parse events file {events_file}: {exc}")

        return _gather_from_iter(input_org, rows)

    # 2. SQLite path
    if db_file is None:
        terminate("You must supply --db-file or --events-file.")

    if not db_file.exists():
        terminate(f"SQLite database not found: {db_file}")

    try:
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT repo_org, repo_name, before, timestamp
                FROM pushes
                WHERE repo_org = ?
                """,
                (input_org,),
            )
            rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        terminate(f"Failed querying SQLite DB {db_file}: {exc}")

    return _gather_from_iter(input_org, rows)


############################################################
# Phase 2: Reporting
############################################################

def generate_github_urls(repos: Dict[str, List[dict]]) -> List[str]:
    urls = []
    for repo_url, commits in repos.items():
        # Extract repo_org and repo_name from repo_url
        # Assuming repo_url is in the format https://github.com/{repo_org}/{repo_name}
        parts = repo_url.split('/')
        repo_org = parts[-2]
        repo_name = parts[-1]

        for commit_info in commits:
            commit_hash = commit_info["before"]
            urls.append(f"https://github.com/{repo_org}/{repo_name}/commit/{commit_hash}.patch")
    return urls

def report(input_org: str, repos: Dict[str, List[dict]]) -> None:
    repo_count = len(repos)
    total_commits = sum(len(v) for v in repos.values())

    print(f"\n{Fore.CYAN}======= Force-Push Summary for {input_org} ======={Style.RESET_ALL}")
    print(f"{Fore.GREEN}Repos impacted : {repo_count}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Total commits  : {total_commits}{Style.RESET_ALL}\n")

    # per-repo counts
    for repo_url, commits in repos.items():
        print(f"{Fore.YELLOW}{repo_url}{Style.RESET_ALL}: {len(commits)} commits")
    print()

    # timeseries histogram (yearly) – include empty years
    counter = Counter(to_year(c["date"]) for commits in repos.values() for c in commits)

    if counter:
        first_year = int(min(counter))
    else:
        first_year = _dt.date.today().year

    current_year = _dt.date.today().year

    print(f"{Fore.CYAN}Histogram:{Style.RESET_ALL}")
    for year in range(first_year, current_year + 1):
        year_key = f"{year:04d}"
        count = counter.get(year_key, 0)
        bar = "▇" * min(count, 40)
        if count > 0:
            print(f" {Fore.GREEN}{year_key}{Style.RESET_ALL} | {bar} {count}")
        else:
            print(f" {year_key} | ")
    print("=================================\n")


############################################################
# Phase 3: Secret scanning
############################################################


def custom_secret_scan(url: str, patterns: List[re.Pattern]) -> List[Tuple[str, str, str, str]]:
    findings = []
    print(f"[>] Scanning URL for secrets: {url}")
    try:
        # Fetch content using curl
        curl_command = ["curl", "-s", url]
        content = run(curl_command)

        # Search for patterns
        for compiled_pattern in patterns:
            matches = compiled_pattern.finditer(content)
            for match in matches:
                value = match.group(0)
                if value:
                    # Strip leading '+' or '-' from the value if it's from a git diff
                    if value.startswith('+') or value.startswith('-'):
                        value = value[1:]
                    context_start = max(0, match.start() - 50)  # Get 50 characters before the match
                    context = content[context_start:match.start()]
                    findings.append((url, compiled_pattern.pattern, context.strip(), value))

    except RunCmdError as err:
        print(f"[!] Failed to fetch URL {url}: {err}")
    except Exception as e:
        print(f"[!] An error occurred during custom secret scan for {url}: {e}")
    return findings

def identify_base_commit(repo_path: Path, since_commit:str) -> str:
    """Identify the base commit for the given repository and since_commit."""    # fetch the since_commit, since our clone process likely missed it
    # note: this fetch will have no blobs, but that's fine b/c 
    # when we invoke trufflehog, it calls git log -p, which will fetch the blobs dynamically
    run(["git", "fetch", "origin", since_commit], cwd=repo_path)
    # get all commits reachable from the since_commit
    output = run(["git", "rev-list", since_commit], cwd=repo_path)
    # working backwards from the since_commit, we need to find the first commit that exists in any branch
    for commit in output.splitlines():
        #remove the newline character
        commit = commit.strip('\n')
        # Check if commit exists in any branch, if it does, we've found the base commit
        if run(["git", "branch", "--contains", commit, "--all"], cwd=repo_path):
            if commit != since_commit:
                return commit
            try:
                # if the commit is the same as the since_commit, we need to go back one commit to scan this commit
                # if there is no commit~1, then since_commit is the base commit and we need "" for trufflehog
                c = run(["git", "rev-list", commit + "~1", "-n", "1"], cwd=repo_path)
                return c.strip('\n')
            except RunCmdError as err: # need to handle 128 git errors
                return ""
        continue
    # if we get here, then the since_commit is not in any branch
    # which means it could be a force push of a whole new tree or similar
    # in this case, we need to scan the entire branch, so we return ""
    # note: The command below might be useful if we find an edge case 
    #       not covered by "" in the future.
    #       c = run(["git", "rev-list", "--max-parents=0", 
    #           since_commit, "-n", "1"], cwd=repo_path)
    #       return c.strip('\n')
    return ""

def scan_commits(repo_user: str, repos: Dict[str, List[dict]], mantra_patterns: List[str]) -> None:
    all_unique_findings = {} # To store unique findings (value -> (url, pattern, context))

    for repo_url, commits in repos.items():
        print(f"\n[>] Scanning repo: {repo_url}")

        commit_counter = 0
        skipped_repo = False

        tmp_dir = tempfile.mkdtemp(prefix="gh-repo-")
        try:
            tmp_path = Path(tmp_dir)
            try:
                # Partial clone with no blobs to save space and for speed
                run(
                    [
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--no-checkout",
                        repo_url + ".git",
                        ".",
                    ],
                    cwd=tmp_path,
                )
            except RunCmdError as err:
                print(f"[!] git clone failed: {err} — skipping this repository")
                skipped_repo = True
                continue

            for c in commits:
                before = c["before"]
                if not _SHA_RE.fullmatch(before):
                    print(f"  • Commit {before} – invalid SHA, skipping")
                    continue
                commit_counter += 1
                print(f"  • Commit {before}")
                try:
                    since_commit = identify_base_commit(tmp_path, before)
                except RunCmdError as err:
                    # If the commit was logged in GH Archive, but not longer exists in the repo network, then it was likely manually removed it.
                    # For more details, see: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository#:~:text=You%20cannot%20remove,rotating%20affected%20credentials.
                    if "fatal: remote error: upload-pack: not our ref" in str(err):
                        print("    This commit was likely manually removed from the repository network  — skipping commit")
                    else:
                        print(f"    fetch/checkout failed: {err} — skipping commit")
                    continue

                # Custom secret scanning
                findings = custom_secret_scan(f"{repo_url}/commit/{before}.patch", mantra_patterns)
                for finding in findings:
                    url, pattern, context, value = finding
                    if value not in all_unique_findings:
                        all_unique_findings[value] = (url, pattern, context)

        finally:
            # Attempt cleanup but suppress ENOTEMPTY race-condition errors
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except OSError:
                print(f"    Error cleaning up temporary directory: {tmp_dir}")
                pass

        if skipped_repo:
            print("[!] Repo skipped due to earlier errors")
        else:
            print(f"[✓] {commit_counter} commits scanned.")

    # Print unique findings
    if all_unique_findings:
        print(f'''\n{Fore.CYAN}======= Unique Secret Findings ======={Style.RESET_ALL}''')
        for value, (url, pattern, context) in all_unique_findings.items():
            print(f"""
{Fore.RED}[!] Secret Found!{Style.RESET_ALL}
    {Fore.YELLOW}URL:     {url}{Style.RESET_ALL}
    {Fore.MAGENTA}Pattern: {pattern}{Style.RESET_ALL}
    {Fore.WHITE}Context: {context}{Style.RESET_ALL}
    {Fore.GREEN}Value:   {value}{Style.RESET_ALL}
----------------------------------------""")

    # Generate URLs for mantra and run mantra
    all_github_urls = generate_github_urls(repos)
    if all_github_urls:
        print("\n[>] Running mantra on generated URLs...")
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_url_file:
            for url in all_github_urls:
                temp_url_file.write(url + "\n")
            temp_file_path = temp_url_file.name
        try:
            cat_proc = subprocess.Popen(["cat", temp_file_path], stdout=subprocess.PIPE)
            run_mantra_cmd = ["/home/kali/go/bin//mantra"]
            mantra_proc = subprocess.run(run_mantra_cmd, stdin=cat_proc.stdout, capture_output=True, text=True, check=True)
            cat_proc.wait()
            # Filter out revert commit messages
            for line in mantra_proc.stdout.splitlines():
                if "This reverts commit" not in line:
                    print(line)
            print("[✓] Mantra scan completed.")
        except RunCmdError as err:
            print(f"[!] Mantra execution failed: {err}")
        except Exception as e:
            print(f"[!] An error occurred during mantra execution: {e}")
        finally:
            os.remove(temp_file_path)


############################################################
# Entry point
############################################################
def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    events_path = Path(args.events_file) if args.events_file else None
    db_path = Path(args.db_file) if args.db_file else None

    # Read mantra patterns
    mantra_patterns_path = Path(__file__).parent / "mantra_patterns_cleaned.txt"
    if not mantra_patterns_path.exists():
        terminate(f"Mantra patterns file not found: {mantra_patterns_path}")
    with mantra_patterns_path.open("r", encoding="utf-8") as f:
        raw_mantra_patterns = [line for line in f] # Keep newlines for multiline processing

    compiled_mantra_patterns = _compile_patterns(raw_mantra_patterns)

    repos = gather_commits(args.input_org, events_path, db_path)
    report(args.input_org, repos)
    
    if args.scan:
        scan_commits(args.input_org, repos, compiled_mantra_patterns)
    else:
        print("[✓] Exiting without scan.")


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect force-push commit events from public GitHub orgs and optionally scan their git diff patches for secrets.",
    )
    parser.add_argument(
        "input_org",
        help="GitHub username or organization to inspect",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Run a trufflehog scan on every force-pushed commit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose / debug logging",
    )
    parser.add_argument(
        "--events-file",
        help="Path to a CSV file containing force-push events. 4 columns: repo_org, repo_name, before, timestamp",
    )
    parser.add_argument(
        "--db-file",
        help="Path to the SQLite database containing force-push events. 4 columns: repo_org, repo_name, before, timestamp",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Ensure required external tools are available early.
    if shutil.which("git") is None:
        terminate(f"Required tool 'git' not found in PATH")
    main()
