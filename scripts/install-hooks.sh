#!/bin/sh
# Install this repo's git hooks. Run once after cloning:
#
#     sh scripts/install-hooks.sh
#
# Hooks are not versioned by git, which is exactly why this exists: the scan
# that keeps private data out of a public repo cannot itself be published into
# your clone. Nothing installs it for you.
#
# It copies rather than symlinks, so a hook keeps working while you edit or
# check out a different branch, and it refuses to clobber a hook you wrote
# yourself unless you pass --force.
set -e

usage() {
    echo "usage: sh scripts/install-hooks.sh [--force]" >&2
    exit 2
}

force=0
case "$1" in
    --force) force=1 ;;
    '')      ;;
    *)       usage ;;
esac

root=$(git rev-parse --show-toplevel)
cd "$root"
hooks_dir=$(git rev-parse --git-path hooks)
mkdir -p "$hooks_dir"

installed=0
for name in pre-commit commit-msg pre-push; do
    src="scripts/hooks/$name"
    dst="$hooks_dir/$name"
    if [ ! -f "$src" ]; then
        echo "install-hooks: missing $src" >&2
        exit 1
    fi
    if [ -e "$dst" ] && [ "$force" = 0 ] && ! cmp -s "$src" "$dst"; then
        # Do not silently overwrite someone's own hook. Silently replacing it
        # would break their workflow in a way that looks like git misbehaving.
        echo "install-hooks: $name differs from the shipped one — leaving it."
        echo "               Re-run with --force to overwrite, or merge by hand:"
        echo "                 diff $dst $src"
        continue
    fi
    cp "$src" "$dst"
    chmod +x "$dst"
    installed=$((installed + 1))
done

echo "install-hooks: $installed hook(s) installed into $hooks_dir"

# The personal-identifier rules are git-ignored, so a fresh clone has none and
# the hooks would run with the generic rules only. Say so — a gate that
# overstates what it checked is worse than no gate, because it is trusted.
if [ ! -f scripts/scrub-rules.local.txt ]; then
    echo ""
    echo "install-hooks: NOTE — scripts/scrub-rules.local.txt does not exist."
    echo "               The hooks will run WITHOUT any personal-identifier rules"
    echo "               (no names, hostnames or handles). Create it — one regex"
    echo "               per line — if you fork this for your own household."
fi

echo ""
echo "Verify with:  python3 scripts/scrub_check.py --selftest"
