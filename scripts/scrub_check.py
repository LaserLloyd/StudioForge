#!/usr/bin/env python3
"""Fail-closed scan for private data in the public tree.

This is the guarantee behind "my data never leaves". It runs six ways:

    python3 scripts/scrub_check.py                    # scan the whole repo
    python3 scripts/scrub_check.py --staged           # pre-commit: the INDEX
    python3 scripts/scrub_check.py --path DIR         # an arbitrary directory
    python3 scripts/scrub_check.py --rev SHA          # pre-push: a commit TREE
    python3 scripts/scrub_check.py --message FILE     # commit-msg: one message
    python3 scripts/scrub_check.py --commit-range A..B  # pre-push: N messages

Install the hooks that use them with `scripts/install-hooks.sh`.

Exit status is 0 only when the tree is clean. Anything suspicious is a hard
failure with the file, line and reason — it never "warns and continues",
because a warning in a publish pipeline is a leak with extra steps.

Design notes
------------
* **Deny by pattern, not by filename.** A rule keyed to `security.yaml` misses
  `security.yaml.bak`; a rule that recognises a PIN hash catches both.
* **Personal identifiers are configurable**, because the maintainer's own name
  and hostnames are exactly what must not ship, and they differ per fork. Set
  them in `scripts/scrub-rules.local.txt` (git-ignored, one regex per line) so
  a fork can add its own without editing this file.
* **False positives are expected and cheap to handle**: add an inline
  `# scrub-ok: <reason>` comment on the offending line. That marker is
  deliberately noisy so it shows up in review.
* **`--staged` reads the INDEX, not the working tree.** Those are different
  files, and the difference is exploitable: `git add secrets.py`, edit the
  secret out of the working copy, commit — the hook reads the clean working
  copy and the index still holds the secret. Every staged read here goes
  through `git show :<path>`.
* **Some rules only apply to files git is actually carrying.** A working tree
  contains a running install's model files, logs and generated output; git
  ignores them, so they cannot be published and flagging them would only teach
  people that this tool cries wolf. Those live in FORBIDDEN_IF_COMMITTED and
  fire when the file is tracked or staged.
* **Commit messages go through the same regexes.** They are published as
  loudly as the code is, and a message is where a path, a hostname or an
  assistant session URL gets pasted without thinking. `--message` and
  `--commit-range` reuse CONTENT_RULES rather than growing a second copy of
  them in shell, because two copies of a deny-list is one stale deny-list.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The verdicts use ✓/✗ and em-dashes. Under a git hook on Windows,
# stdout is a cp1252 pipe and printing them raises UnicodeEncodeError -- the
# hook then fails the COMMIT with a codec traceback on a perfectly clean tree.
# The gate must never be the thing that breaks; degrade the glyphs, not the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - exotic streams
        pass

REPO = Path(__file__).resolve().parent.parent

# Files we never scan (binary or vendored) — but see FORBIDDEN_PATHS below,
# which still bars whole categories from existing at all.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".woff", ".woff2",
    ".ttf", ".otf", ".mp4", ".webm", ".mp3", ".zip", ".gz", ".tar", ".whl",
    ".pdf", ".lock",
}
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "vendor", "htmlcov", "dist", "build",
}

# Paths that must not exist in the public tree at all, whatever they contain.
FORBIDDEN_PATHS = [
    (re.compile(r"(^|/)chats\.db"),            "the live chat database"),
    (re.compile(r"\.db(-wal|-shm)?$"),         "a SQLite database"),
    (re.compile(r"(^|/)security\.yaml"),       "the PIN/credential store"),
    (re.compile(r"(^|/)trusted-devices\.yaml"), "remembered-device tokens"),
    (re.compile(r"(^|/)config\.yaml$"),        "an operator's bot roster"),
    (re.compile(r"(^|/)RECOVERY-CODE"),        "a recovery code"),
    (re.compile(r"(^|/)gateway-mirror\.json"), "mirror state with session ids"),
    # `.env` AND `.env.production`, `.env.local`, … — the gitignore covers the
    # whole `.env.*` family and a rule that only knew the bare name would let
    # every suffixed one through. `.env.example` is the shipped template.
    (re.compile(r"(^|/)\.env(?:$|\.(?!example$))"),
     "a real environment file"),
    (re.compile(r"(^|/)(media|files|backups|logs)/"), "a runtime data directory"),
    # moods dirs are per-bot since the per-bot pool port: reactions/moods-<bot>/
    (re.compile(r"(^|/)reactions/(pool|spent|moods)(-[A-Za-z0-9_-]+)?/"), "generated reaction blobs"),
    (re.compile(r"\.bak(-|$)"),                "a stray backup file"),
]

# Paths that are fine to HAVE (a working install writes them into the tree) but
# must never be committed. Checked only when git is actually carrying the file
# — tracked in a full scan, staged in --staged.
#
# These are all images, and images are the blind spot: SKIP_SUFFIXES stops the
# content scan on them, so a photograph of somebody's family sails through
# every rule above. Nothing else in this file would catch them.
FORBIDDEN_IF_COMMITTED = [
    (re.compile(r"(^|/)reactions/(pack|builtin)/"),
     "a reaction image from a real install"),
]

# Content patterns. Each is (regex, human explanation).
CONTENT_RULES: list[tuple[re.Pattern, str]] = [
    # --- Credentials -------------------------------------------------------
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),        "an Anthropic API key"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"),               "an OpenAI-style API key"),
    (re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}"), "a Stripe secret key"),
    # GitHub's whole token family shares one prefix scheme; ghp_ alone missed
    # OAuth (gho_), user-to-server (ghu_), server-to-server (ghs_), refresh
    # (ghr_) and the fine-grained PATs that are what people paste today.
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),         "a GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),       "a fine-grained GitHub PAT"),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}"),         "a GitLab personal access token"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),       "a Slack token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                "an AWS access key id"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),           "a Google API key"),
    (re.compile(r"\bBSA[A-Za-z0-9_\-]{20,}"),            "a Brave API key"),
    # A literal Bearer token. Placeholders are how docs SHOULD write this, so
    # anything containing <, $, {, or the word TOKEN/KEY is left alone.
    (re.compile(r"(?i)\bBearer\s+(?![A-Za-z0-9_]*(?:TOKEN|KEY|SECRET|HERE)\b)"
                r"[A-Za-z0-9._\-]{24,}"),                "a literal Bearer token"),
    # A JWT: three base64url segments, the first decoding from '{"'. Two dots
    # are required — one-dot matches were hitting ordinary dotted identifiers.
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
     "a JWT (it decodes to whoever it was issued for)"),
    # PEM headers vary by algorithm AND suffix: "OPENSSH PRIVATE KEY",
    # "PGP PRIVATE KEY BLOCK". Anchoring on the tail dashes missed the latter.
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY[A-Z ]*-----"),  "a private key"),
    (re.compile(r"(?i)\b(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*"
                r"['\"][^'\"\s]{12,}['\"]"),             "a hard-coded credential"),
    (re.compile(r"\bpbkdf2_sha256\$|\$argon2[a-z]{1,2}\$"), "a password hash"),

    # --- AI-assistant provenance ------------------------------------------
    # The public log is a human record. Session URLs additionally point at a
    # private transcript of this machine, so they are a leak, not just noise.
    (re.compile(r"claude\.ai/code/session[_/][A-Za-z0-9_\-]+"),
     "a link to a private assistant session transcript"),
    (re.compile(r"(?im)^\s*Claude-Session\s*:"),
     "a Claude-Session trailer (no assistant trailers in the public log)"),
    (re.compile(r"(?im)^\s*Co-Authored-By\s*:.*\b(Claude|Anthropic|Copilot|Cursor)\b"),
     "an AI-assistant Co-Authored-By trailer"),

    # --- Personal / host identifiers --------------------------------------
    (re.compile(r"/(var/)?home/"
                # Documentation placeholders are exactly what these paths SHOULD
                # be — exempt them by name, or every deploy guide trips the tool
                # and people learn to ignore it.
                r"(?!<|\$|\{|USER\b|user\b|username\b|youruser\b|your-user\b"
                r"|you\b|me\b|app\b|pi\b|node\b|dispatch\b|example\b"
                # `[[media:/home/secret.png]]` in backend/app/main.py: a made-up
                # path in a docstring about rejecting made-up paths.
                r"|secret\b|runner\b|ubuntu\b)"
                # No trailing "/" required: a bare home path at the end of a
                # sentence discloses the same username as one with a file after
                # it, and requiring the slash let every bare one through.
                r"[a-z][a-z0-9_-]{1,31}(?![a-z0-9_-])"),
     "an absolute path inside someone's home directory"),
    # This project ships a Windows app: C:\\Users\\<name> discloses the operator
    # exactly as /home/<user> does. Docs placeholders are exempt by name.
    (re.compile(r"(?i)\b[A-Z]:\\+Users\\+"
                r"(?!<|\$|%|\{|USER\b|user\b|username\b|youruser\b|your-user\b"
                r"|you\b|me\b|example\b|Public\b|Default\b|All Users\b)"
                r"[A-Za-z][A-Za-z0-9._-]{1,31}"),
     "an absolute path inside someone's Windows user profile"),

    # A per-user systemd/XDG runtime path carries the operator's uid.
    (re.compile(r"/run/user/(?!<|\$|\{|UID\b|uid\b)\d+"),
     "a uid-specific runtime path (write /run/user/$UID)"),
    # Any Tailscale MagicDNS name, not just the tail<hex> form: a tailnet may
    # use a custom domain and the host label alone identifies the machine.
    # `example.ts.net` is the RFC 2606 style fake used by this repo's tests.
    (re.compile(r"(?<!example)\.ts\.net\b"),
     "a real Tailscale MagicDNS hostname (use example.ts.net in docs)"),
    (re.compile(r"\btail[0-9a-f]{6,}\b"),
     "a Tailscale tailnet id"),
    (re.compile(r"[A-Za-z0-9._%+-]+"
                # noreply@ / no-reply@ are unroutable by convention and are what
                # a git author line SHOULD say, so they are never a disclosure.
                r"(?<!\bnoreply)(?<!\bno-reply)@"
                r"(?!example\.(com|org|net)\b|users\.noreply\.github\.com\b)"
                r"[A-Za-z0-9.-]+\."
                # .local/.internal/.lan/.home are SSH and mDNS targets in these
                # docs, never mailboxes.
                r"(?!local\b|internal\b|lan\b|home\b|arpa\b|invalid\b|test\b)"
                r"[A-Za-z]{2,}"),
     "an email address (use example.com in docs)"),
]

# The marker in every comment syntax this repo actually contains: shell/Python,
# C/JS block and line, HTML/Markdown, SQL (the design docs carry DDL), and
# Windows batch (`rem` / `::`) — this project ships .bat launchers, and a
# launcher is exactly the kind of file that legitimately names a local path.
OK_MARKER = re.compile(r"(?:#|//|--|::)\s*scrub-ok\b"
                       r"|(?:/\*|<!--)\s*scrub-ok\b"
                       r"|(?i:\brem)\s+scrub-ok\b")


def load_local_rules(path: Path | None = None) -> list[tuple[re.Pattern, str]]:
    """Per-fork identifiers: the maintainer's name, hostnames, handles.

    Kept out of the repo on purpose — publishing the list of words that must
    never be published is its own small leak.

    ``path`` overrides the location, which the selftest uses to drive these
    rules with an INVENTED identifier: proving the LICENCE exception needs a
    name the local rules actually match, and writing the real one into a
    fixture is the leak this whole file exists to prevent.
    """
    path = path or REPO / "scripts" / "scrub-rules.local.txt"
    rules = []
    if not path.exists():
        if getattr(load_local_rules, "_noted", False):
            return rules
        load_local_rules._noted = True      # once per run, not once per commit scanned
        # SAY SO. The file is git-ignored on purpose, which means CI — and any
        # contributor's clone — runs this check with NO personal identifiers
        # loaded at all. Staying quiet made the CI step print "clean" and read
        # as "no private data", when what it actually verified was the generic
        # patterns only. A gate that overstates what it checked is worse than
        # no gate, because it is trusted.
        print("scrub_check: NOTE — scripts/scrub-rules.local.txt not found, so "
              "personal-identifier rules are NOT loaded.\n"
              "            Generic rules (paths, keys, emails, IPs) still ran. "
              "On the maintainer's machine that file exists;\n"
              "            in CI it never does, so this run cannot catch a name "
              "or hostname.", file=sys.stderr)
        return rules
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rules.append((re.compile(line, re.IGNORECASE), LOCAL_RULE_WHY))
        except re.error as e:
            print(f"scrub_check: bad regex in scrub-rules.local.txt: {line!r} ({e})",
                  file=sys.stderr)
            sys.exit(2)
    return rules


def local_rule_summary() -> tuple[int, str]:
    """How many private-identifier rules this run actually loaded, in words.

    A "clean" verdict is worth exactly what the rule set behind it was, and
    `scripts/scrub-rules.local.txt` is git-ignored on purpose — so a fresh
    clone, or CI, scans with the GENERIC rules only and still prints clean.
    Every verdict now states the count, so a clean line can never be read as
    more than it is. `--require-local-rules` turns that from a caption into a
    gate.
    """
    n = len(load_local_rules())
    if n:
        return n, (f"{n} private identifier rule(s) loaded from "
                   f"scripts/scrub-rules.local.txt")
    return 0, ("0 private identifier rules — scripts/scrub-rules.local.txt is "
               "absent, so GENERIC rules only")


def _skipped(rel: str) -> bool:
    return any(part in SKIP_DIRS for part in Path(rel).parts)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git with a FIXED argv (never a shell string) in `root`."""
    return subprocess.run(["git", *args], cwd=root,
                          capture_output=True, text=True, check=False)


def git_root(start: Path) -> Path | None:
    r = _git(start, "rev-parse", "--show-toplevel")
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def tracked_files(repo_root: Path) -> set[str]:
    """Paths git is carrying, relative to the repo root ('' if not a repo)."""
    r = _git(repo_root, "ls-files", "-z")
    if r.returncode != 0:
        return set()
    return {p for p in r.stdout.split("\0") if p}


def staged_files(repo_root: Path) -> list[str]:
    """Paths staged for commit, relative to the REPO ROOT.

    git reports repo-root-relative paths whatever directory it is invoked
    from. Joining them onto an arbitrary --path was how `--staged --path
    backend` ended up scanning `backend/backend/app/…`, i.e. nothing.
    """
    r = _git(repo_root, "diff", "--cached", "--name-only", "--diff-filter=ACM")
    return [p for p in r.stdout.split("\n") if p.strip()]


def staged_text(repo_root: Path, rel: str) -> str | None:
    """The STAGED content of a file, or None if it is binary/unreadable.

    `git show :<path>` reads the index. Reading the working copy instead is a
    hole rather than a shortcut: stage a file containing a credential, delete
    the credential from the working copy, and the hook passes while the commit
    still carries the secret.
    """
    r = subprocess.run(["git", "show", f":{rel}"], cwd=repo_root,
                       capture_output=True, check=False)
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def rev_files(repo_root: Path, rev: str) -> list[str]:
    """Every path a COMMIT carries, relative to the repo root.

    `git ls-tree -r` is the tree the push would publish. The working tree is
    not: a pre-push hook that scanned the checkout would pass while the commit
    being pushed still carried a secret the author had since deleted — the same
    index-vs-working-tree hole --staged exists to close, one step further out.
    """
    r = _git(repo_root, "ls-tree", "-r", "-z", "--name-only", rev)
    if r.returncode != 0:
        return []
    return [p for p in r.stdout.split("\0") if p]


def rev_text(repo_root: Path, rev: str, rel: str) -> str | None:
    """The content of one file AS OF a commit, or None if binary/unreadable."""
    r = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=repo_root,
                       capture_output=True, check=False)
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


#: The one place a personal name is published ON PURPOSE.
#:
#: MIT requires the copyright notice to survive verbatim, and naming a
#: copyright holder is a deliberate public authorship statement — the opposite
#: of a leak. The maintainer's legal name is nevertheless a local private
#: identifier everywhere else in the tree, and must stay one: a home path, a
#: hostname or a commit message that names him is still a finding.
#:
#: So the exception is as narrow as it can be made:
#:   * only in a file actually named LICENSE / LICENCE / COPYING,
#:   * only on the line that IS the copyright notice, and
#:   * only against the LOCAL private-identifier rules.
#: Every generic rule — credentials, hosts, IPs, emails, paths — still applies
#: to that line and to the rest of the file.
LICENCE_FILENAMES = {"LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"}
COPYRIGHT_NOTICE = re.compile(
    r"^\s*Copyright\s*(?:\(c\)|©)?\s*\d{4}(?:\s*[-–]\s*\d{4})?\s+\S", re.IGNORECASE)
#: The `why` string load_local_rules attaches to every local rule.
LOCAL_RULE_WHY = "a local private identifier"


def _is_copyright_notice(rel: str, line: str) -> bool:
    return Path(rel).name in LICENCE_FILENAMES and bool(COPYRIGHT_NOTICE.match(line))


HEURISTIC_IN_FIXTURES = {
    "a hard-coded credential",
    "a real Tailscale MagicDNS hostname (use example.ts.net in docs)",
}
FIXTURE_PATH = re.compile(r"(^|/)tests?/|(^|/)scripts/e2e_")


def scan_text(text: str, label: str) -> list[str]:
    """Run the CONTENT rules over free text (a commit message, not a file).

    Path rules make no sense here; everything else does, and sharing the list
    is the point — a second copy of these regexes in shell is a second copy
    that drifts.
    """
    problems: list[str] = []
    rules = CONTENT_RULES + load_local_rules()
    for lineno, line in enumerate(text.splitlines(), 1):
        # git strips its own comment lines before the message is stored, so
        # scanning them would reject commits over the template's own text.
        if line.startswith("#"):
            continue
        if OK_MARKER.search(line):
            continue
        for pattern, why in rules:
            m = pattern.search(line)
            if m:
                snippet = m.group(0)
                if len(snippet) > 60:
                    snippet = snippet[:57] + "\u2026"
                problems.append(f"{label}:{lineno}: looks like {why} \u2192 {snippet!r}")
                break
    return problems


def scan_commit_range(repo_root: Path, spec: str) -> list[str]:
    """Scan the message of every commit named by a rev-list SPEC.

    SPEC is split on whitespace and handed to git as separate argv words, so
    `HEAD~3..HEAD` and `abc123 --not --remotes=origin` both work. The second
    form is what a pre-push hook needs when the remote ref is all-zeroes (a
    brand-new branch): there is no remote sha to diff against, and walking the
    whole history instead would scan — and reject — every commit ever made.
    """
    args = [w for w in spec.split() if w]
    if not args:
        return []
    r = _git(repo_root, "log", "--no-merges", "--format=%H%x1f%B%x1e", *args)
    if r.returncode != 0:
        return [f"commit-range {spec!r}: git log failed \u2014 "
                f"{r.stderr.strip() or 'unknown error'}"]
    problems: list[str] = []
    for record in r.stdout.split("\x1e"):
        record = record.strip("\n")
        if not record.strip():
            continue
        sha, _, body = record.partition("\x1f")
        problems += scan_text(body, f"commit {sha[:12]}")
    return problems


def scan(root: Path, staged: bool = False, rev: str | None = None) -> list[str]:
    """Scan a tree, the index, or a commit, and return every problem found.

    Three sources, one rule set:
      * default \u2014 the working tree under `root`
      * `staged`  \u2014 the INDEX (pre-commit)
      * `rev`     \u2014 the tree of a commit (pre-push)

    In the last two `root` is only used to LOCATE the repository; the scan
    itself is git content, reported with repo-root-relative paths, which is
    what git prints and what the committer needs to act on.
    """
    problems: list[str] = []
    rules = CONTENT_RULES + load_local_rules()

    repo_root = git_root(root)
    if (staged or rev) and repo_root is None:
        mode = "--staged" if staged else "--rev"
        return [f"{root}: {mode} needs a git repository, and this is not one"]

    # Which files git is carrying. In staged mode everything scanned is by
    # definition on its way into a commit, so the committed-only rules always
    # apply; in a full scan they apply to tracked files only.
    carried: set[str] = set()
    if not staged and not rev and repo_root is not None:
        carried = tracked_files(repo_root)

    if rev:
        assert repo_root is not None
        # Everything in a commit is, by definition, published by the push.
        entries = [(rel, repo_root) for rel in rev_files(repo_root, rev)]
    elif staged:
        assert repo_root is not None
        # A --path narrows the staged set instead of being joined onto it.
        try:
            prefix = root.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            prefix = ""
        prefix = "" if prefix in ("", ".") else prefix + "/"
        entries = [(rel, repo_root) for rel in staged_files(repo_root)
                   if rel.startswith(prefix)]
    else:
        entries = [(p.relative_to(root).as_posix(), root)
                   for p in root.rglob("*") if p.is_file()]

    for rel, base in entries:
        if _skipped(rel):
            # SKIP_DIRS applies in BOTH modes. It did not, once: --staged
            # returned git's list verbatim, so vendored third-party bundles
            # were content-scanned only at commit time — and the pre-commit
            # hook duly blocked a commit over the upstream author's email in a
            # minified library's license header. A rule that fires on code
            # nobody in this repo wrote is a rule people learn to bypass.
            continue
        # The rules file necessarily CONTAINS the words it bans, and is
        # git-ignored so it can never be published.
        if rel == "scripts/scrub-rules.local.txt":
            continue

        for pattern, why in FORBIDDEN_PATHS:
            if pattern.search(rel):
                problems.append(f"{rel}: must not exist in a public tree — {why}")
                break

        if staged or rev or rel in carried:
            for pattern, why in FORBIDDEN_IF_COMMITTED:
                if pattern.search(rel):
                    where = ("staged for commit" if staged
                             else f"committed in {rev}" if rev
                             else "tracked by git")
                    problems.append(
                        f"{rel}: {where}, and must not be — {why}")
                    break

        if Path(rel).suffix.lower() in SKIP_SUFFIXES:
            continue

        if rev:
            text = rev_text(base, rev, rel)
        elif staged:
            text = staged_text(base, rel)
        else:
            path = base / rel
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                text = None   # binary or unreadable: path rules above still applied
        if text is None:
            continue

        in_fixtures = bool(FIXTURE_PATH.search(rel))
        for lineno, line in enumerate(text.splitlines(), 1):
            if OK_MARKER.search(line):
                continue
            licence_notice = _is_copyright_notice(rel, line)
            for pattern, why in rules:
                if in_fixtures and why in HEURISTIC_IN_FIXTURES:
                    continue
                if licence_notice and why == LOCAL_RULE_WHY:
                    continue  # a copyright holder is published on purpose
                m = pattern.search(line)
                if m:
                    snippet = m.group(0)
                    if len(snippet) > 60:
                        snippet = snippet[:57] + "…"
                    problems.append(f"{rel}:{lineno}: looks like {why} → {snippet!r}")
                    break
    return problems


def selftest() -> int:
    """Prove --staged reads the index. Builds a throwaway repo; touches nothing.

    Written because the bug it guards against is invisible from the outside:
    the hook printed "clean", the commit went through, and the secret was in
    it. A test that stages one thing and leaves another on disk is the only
    way to tell the two reads apart.
    """
    import tempfile

    failures: list[str] = []
    secret = 'api_key = "s3cr3t-not-a-real-key-000"'  # scrub-ok: fixture; must LOOK like a key

    with tempfile.TemporaryDirectory(prefix="scrub-selftest-") as tmp:
        root = Path(tmp)
        _git(root, "init", "-q")
        (root / "backend").mkdir()
        leak = root / "backend" / "leak.py"
        leak.write_text(secret + "\n", encoding="utf-8")
        _git(root, "add", "backend/leak.py")
        # The exploit: tidy the WORKING copy after staging the dirty one.
        leak.write_text("api_key = os.environ['API_KEY']\n", encoding="utf-8")

        if not any("backend/leak.py" in p for p in scan(root, staged=True)):
            failures.append("--staged missed a credential that is in the index "
                            "but not in the working tree")
        if any("leak.py" in p for p in scan(root)):
            failures.append("the working-tree scan flagged a file whose credential "
                            "was already removed (the two reads are not distinct)")
        if not any("backend/leak.py" in p for p in scan(root / "backend", staged=True)):
            failures.append("--staged --path <subdir> built the wrong paths")

        (root / "docs").mkdir()
        if any("leak.py" in p for p in scan(root / "docs", staged=True)):
            failures.append("--staged --path <other subdir> scanned outside its subtree")

        # Windows user paths: this project ships a Windows app, so C:\\Users\\<name>
        # is the same disclosure as /home/<user>. Placeholders must NOT fire.
        winleak = root / "docs" / "win.md"
        winleak.write_text("run C:\\Users\\jsmith\\StudioForge\\run.bat\n", encoding="utf-8")  # scrub-ok: selftest fixture
        _git(root, "add", "docs/win.md")
        if not any("win.md" in p for p in scan(root, staged=True)):
            failures.append("a Windows user-profile path was not flagged")
        winleak.write_text("run C:\\Users\\<you>\\StudioForge\\run.bat\n", encoding="utf-8")
        _git(root, "add", "docs/win.md")
        if any("win.md" in p for p in scan(root, staged=True)):
            failures.append("a Windows PLACEHOLDER path was flagged — that false "
                            "positive is what gets this tool switched off")
        _git(root, "rm", "-q", "-f", "docs/win.md")


        # --- rev mode: the commit, not the checkout --------------------------
        # Same hole as --staged, one step out: a pre-push hook that reads the
        # working tree passes while the commit being pushed still carries the
        # secret. Commit the dirty file, clean the checkout, then scan the sha.
        (root / "backend" / "hist.py").write_text(secret + "\n", encoding="utf-8")
        _git(root, "add", "backend/hist.py")
        _git(root, "-c", "user.email=t@example.com", "-c", "user.name=t",
             "commit", "-q", "-m", "add hist")
        sha = _git(root, "rev-parse", "HEAD").stdout.strip()
        (root / "backend" / "hist.py").write_text(
            "api_key = os.environ['API_KEY']\n", encoding="utf-8")
        if not any("backend/hist.py" in p for p in scan(root, rev=sha)):
            failures.append("--rev missed a credential that is in the commit "
                            "but not in the working tree")

        # --- the LICENCE copyright-line exception ---------------------------
        # A copyright holder is named on purpose; a hostname or a credential is
        # not. Driven with an INVENTED private identifier (the real one must
        # never appear in a fixture) by pointing load_local_rules at a
        # throwaway rules file for the duration.
        rules_probe = root / "invented-rules.txt"
        rules_probe.write_text(r"\bwintermute\b" + "\n", encoding="utf-8")
        real_loader = load_local_rules
        globals()["load_local_rules"] = lambda path=None: real_loader(rules_probe)
        try:
            (root / "LICENSE").write_text(
                "MIT License\n\nCopyright (c) 2026 Wintermute\n", encoding="utf-8")
            _git(root, "add", "LICENSE")
            if any("LICENSE" in p for p in scan(root, staged=True)):
                failures.append("the LICENCE copyright notice was rejected — MIT "
                                "requires it verbatim, so this makes attribution "
                                "impossible")

            # ...but the same name anywhere else in the same file is a finding.
            (root / "LICENSE").write_text(
                "MIT License\n\nCopyright (c) 2026 Someone\n\nContact Wintermute.\n",
                encoding="utf-8")
            _git(root, "add", "LICENSE")
            if not any("LICENSE" in p for p in scan(root, staged=True)):
                failures.append("the copyright exception leaked past the notice "
                                "line: anything in LICENSE would be waved through")

            # ...and in any other file.
            (root / "docs").mkdir(exist_ok=True)   # git rm prunes empty dirs
            (root / "docs" / "credits.md").write_text(
                "Copyright (c) 2026 Wintermute\n", encoding="utf-8")
            _git(root, "add", "docs/credits.md")
            if not any("credits.md" in p for p in scan(root, staged=True)):
                failures.append("the copyright exception leaked outside LICENSE: "
                                "a name in any file would be waved through")
            _git(root, "rm", "-q", "-f", "docs/credits.md")

            # A credential ON the notice line is still a finding: only the
            # LOCAL identifier rules are relaxed there, never the generic ones.
            (root / "LICENSE").write_text(
                f"MIT License\n\nCopyright (c) 2026 Wintermute, {secret}\n",
                encoding="utf-8")
            _git(root, "add", "LICENSE")
            if not any("LICENSE" in p for p in scan(root, staged=True)):
                failures.append("a credential on the LICENCE copyright line was "
                                "not flagged — the exception must only relax the "
                                "local identifier rules")
        finally:
            globals()["load_local_rules"] = real_loader
            _git(root, "rm", "-q", "-f", "LICENSE")

        # --- content rule fixtures ------------------------------------------
        # Every rule gets a POSITIVE (must fire) and a NEGATIVE (must not).
        # The negatives are the ones that matter: a scanner that flags
        # `/home/you` in a deploy guide, or the documented CGNAT range, gets
        # turned off within a week and then catches nothing at all.
        positives = {
            "tailscale-magicdns": "https://mybox.tailc0ffee.ts.net:8443/",  # scrub-ok: selftest fixture
            "tailnet-id":         "tailnet is tailc0ffee00",  # scrub-ok: selftest fixture
            "home-path-bare":     "cd /var/home/operator",  # scrub-ok: selftest fixture
            "home-path-slash":    "logs in /home/operator/.local/share",  # scrub-ok: selftest fixture
            "runtime-uid":        "XDG_RUNTIME_DIR=/run/user/1000",  # scrub-ok: selftest fixture
            "email":              "contact me at person@somecompany.co.uk",  # scrub-ok: selftest fixture
            "openai-key":         "sk-abcdefghij0123456789abcdefghij",  # scrub-ok: selftest fixture
            "stripe-key":         "sk_live_abcdefghij0123456789",  # scrub-ok: selftest fixture
            "github-pat":         "github_pat_11ABCDEFG0abcdefghijkl",  # scrub-ok: selftest fixture
            "github-oauth":       "gho_16CharsAndMoreToPassTheLengthGate12",  # scrub-ok: selftest fixture
            "gitlab-pat":         "glpat-abcdefghij0123456789",  # scrub-ok: selftest fixture
            "slack-token":        "xoxb-1234567890-abcdefghijkl",  # scrub-ok: selftest fixture
            "google-key":         "AIzaSyA0123456789abcdefghijklmnopqrstuvw",  # scrub-ok: selftest fixture
            "bearer-token":       "Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345",  # scrub-ok: selftest fixture
            "jwt":                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",  # scrub-ok: selftest fixture
            "pem-pgp":            "-----BEGIN PGP PRIVATE KEY BLOCK-----",  # scrub-ok: selftest fixture
            "pem-openssh":        "-----BEGIN OPENSSH PRIVATE KEY-----",  # scrub-ok: selftest fixture
            "session-url":        "see https://claude.ai/code/session_01ABCdef",  # scrub-ok: selftest fixture
            "session-trailer":    "Claude-Session: https://example.invalid/x",  # scrub-ok: selftest fixture
            "ai-coauthor":        "Co-Authored-By: Claude <noreply@anthropic.com>",  # scrub-ok: selftest fixture
        }
        negatives = {
            "docs-tailnet":       "assert host == 'chat-host.example.ts.net'",
            "cgnat-range":        "Tailscale hands out 100.64.0.0/10 addresses",
            "cgnat-range-spaced": "the 100.64.0.0 / 10 block",
            "home-placeholder-1": "mount /home/you/.openclaw into the container",
            "home-placeholder-2": "ReadOnlyPaths=/home/youruser/.openclaw",
            "home-placeholder-3": "home is /home/app in the container",
            "home-placeholder-4": "cd ~   # /home/<you> — the Linux disk",
            "home-placeholder-5": "reject a typed path like /home/secret.png",
            "home-placeholder-6": "export DATA=/home/$USER/.local/share",
            "runtime-uid-ph":     "XDG_RUNTIME_DIR=/run/user/$UID",
            "email-example":      "author: nobody@example.com",
            "email-noreply":      "1234+user@users.noreply.github.com",
            "email-noreply-bare": "git config user.email noreply@dispatch.chat",
            "ssh-host":           "scp file dispatch@chat.local:/srv",
            "bearer-placeholder": "curl -H 'Authorization: Bearer $API_TOKEN'",
            "bearer-placeholder2": "Authorization: Bearer YOUR_TOKEN_HERE",
            "human-coauthor":     "Co-Authored-By: A Contributor <a@example.com>",
            "word-tailed":        "the log is tailed by the mirror poller",
            "sk-word":            "the task sk-eleton is not a key",
        }
        fixtures = root / "fixtures"
        fixtures.mkdir()
        for name, body in {**positives, **negatives}.items():
            (fixtures / f"{name}.txt").write_text(body + "\n", encoding="utf-8")
        hits = {p.split(":", 1)[0].split("/")[-1].removesuffix(".txt")
                for p in scan(fixtures)}
        for name in positives:
            if name not in hits:
                failures.append(f"content rule fixture {name!r} was NOT flagged")
        for name in negatives:
            if name in hits:
                failures.append(f"FALSE POSITIVE: legitimate fixture {name!r} "
                                f"was flagged")

        # A scrub-ok marker still suppresses a genuine hit, in every mode.
        (fixtures / "marked.txt").write_text(
            "gateway at 100.100.10.7  # scrub-ok: fixture\n", encoding="utf-8")
        if any("marked.txt" in p for p in scan(fixtures)):
            failures.append("a scrub-ok marker did not suppress the hit")

        # The .bat launchers this project ships can only carry the marker in
        # batch comment syntax, so `rem` and `::` must suppress it too.
        (fixtures / "marked.bat").write_text(
            'set "H=100.100.10.7"  rem scrub-ok: fixture\n'
            ":: scrub-ok: fixture — gateway at 100.100.10.8\n",
            encoding="utf-8")
        if any("marked.bat" in p for p in scan(fixtures)):
            failures.append("a batch scrub-ok marker did not suppress the hit")

        # --- commit-message fixtures ----------------------------------------
        for name, body in positives.items():
            if not scan_text(body, "msg"):
                failures.append(f"commit-message scan missed {name!r}")
        for name, body in negatives.items():
            if scan_text(body, "msg"):
                failures.append(f"commit-message scan false-positived on {name!r}")
        if scan_text("feat(scrub): add the pre-push hook\n\nNo secrets here.\n", "msg"):
            failures.append("a plain conventional commit message was rejected")
        # git strips its own comment lines, so scanning them would reject a
        # commit over the text of the template git itself wrote.
        git_comment_block = ("feat: x\n"
                             "# On branch main\n"
                             "# Author: someone@corp.example.net\n")  # scrub-ok: fixture
        if scan_text(git_comment_block, "msg"):
            failures.append("the git comment block was scanned (it is stripped "
                            "before the message is stored)")
        if not scan_commit_range(root, "HEAD~0..HEAD") == []:
            failures.append("an empty commit range should be clean, not an error")

    print("scrub_check --selftest: index-vs-working-tree, commit-vs-working-tree, "
          "--path narrowing,\n"
          "                        committed-only image rules, "
          f"{len(positives)} positive + {len(negatives)} negative content "
          "fixtures,\n                        commit-message scanning")
    if failures:
        print("\nFAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    # State the rule set the OK stands on: the local file is git-ignored, so
    # "OK" on a fresh clone means the GENERIC rules passed and nothing more.
    print(f"OK ({local_rule_summary()[1]})")
    return 0


def _report(problems: list[str], scanned: str) -> int:
    """Print the verdict. One place, so every mode fails the same loud way."""
    if problems:
        print(f"\n\u2717 scrub_check: {len(problems)} problem(s) \u2014 refusing to call this clean.\n",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("\nFix it, or if it is genuinely a false positive add an inline"
              "\n`scrub-ok` comment on that line explaining why."
              "\n(A commit message cannot carry a marker \u2014 reword it instead.)\n",
              file=sys.stderr)
        return 1
    print(f"\u2713 scrub_check: clean ({scanned}; {local_rule_summary()[1]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", action="store_true",
                    help="scan the staged content of files staged for commit "
                         "(pre-commit hook) — reads the index, not the working tree")
    ap.add_argument("--path", type=Path, default=REPO,
                    help="directory to scan; with --staged, restrict the staged "
                         "set to this subdirectory (default: the repo root)")
    ap.add_argument("--rev", metavar="COMMIT",
                    help="scan the TREE OF A COMMIT instead of the working tree "
                         "(pre-push) \u2014 reads `git ls-tree`/`git show <rev>:<path>`")
    ap.add_argument("--message", metavar="FILE",
                    help="scan a commit message file with the content rules "
                         "(commit-msg hook); '-' reads stdin")
    ap.add_argument("--commit-range", metavar="SPEC",
                    help="scan the message of every commit in SPEC, a rev-list "
                         "spec such as 'HEAD~3..HEAD' or "
                         "'<sha> --not --remotes=origin' (pre-push)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify --staged really reads the index, then exit")
    # siftforge's copy spells it --self-test; accept both so a runbook written
    # against either repo works here rather than dying on an unknown flag.
    ap.add_argument("--self-test", dest="selftest", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--require-local-rules", action="store_true",
                    help="fail (exit 2) unless scripts/scrub-rules.local.txt "
                         "loaded at least one private-identifier rule — for CI "
                         "or a release gate, where a generic-rules-only 'clean' "
                         "must not pass as a privacy check")
    args = ap.parse_args()

    if args.require_local_rules and not local_rule_summary()[0]:
        print("scrub_check: --require-local-rules was given, but "
              "scripts/scrub-rules.local.txt loaded 0 rules.\n"
              "            This run would have checked GENERIC patterns only "
              "and still printed 'clean'.\n"
              "            Refusing: a verdict that overstates what it checked "
              "is worse than no verdict.", file=sys.stderr)
        return 2

    if args.selftest:
        return selftest()

    root = args.path.resolve()
    if not root.is_dir():
        # `--path CONTRIBUTING.md` used to die inside subprocess with
        # NotADirectoryError from git, which reads as "the tool is broken"
        # rather than "you passed the wrong thing".
        print(f"scrub_check: --path takes a DIRECTORY; {args.path} is not one.",
              file=sys.stderr)
        return 2

    # Message modes scan text, not files: report them on their own so a hook
    # can run just this check without also walking a tree.
    if args.message or args.commit_range:
        problems = []
        scanned_desc = []
        if args.message:
            if args.message == "-":
                text = sys.stdin.read()
                label = "commit message"
            else:
                text = Path(args.message).read_text(encoding="utf-8",
                                                    errors="replace")
                label = args.message
            problems += scan_text(text, label)
            scanned_desc.append("commit message")
        if args.commit_range:
            repo_root = git_root(root)
            if repo_root is None:
                problems.append(f"{root}: --commit-range needs a git repository")
            else:
                problems += scan_commit_range(repo_root, args.commit_range)
            scanned_desc.append(f"commit messages in {args.commit_range!r}")
        return _report(problems, " + ".join(scanned_desc))

    problems = scan(root, staged=args.staged, rev=args.rev)
    scanned = ("staged files" if args.staged
               else f"commit {args.rev}" if args.rev
               else str(root))
    return _report(problems, scanned)


if __name__ == "__main__":
    sys.exit(main())
