#!/usr/bin/env bash
# Checks that no personal path, e-mail or credential is in the repository (git-tracked files or, before the first
# commit, every file outside .git/.venv), and that the files named in README.md and docs/REPRODUCE.md exist.
set -u
cd "$(dirname "$0")/.."
status=0

if git rev-parse --git-dir >/dev/null 2>&1 && [ -n "$(git ls-files)" ]; then files=$(git ls-files); else
  files=$(find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './checkpoints/*' -not -name '*.pyc'); fi

# This script contains the patterns it looks for, so it is excluded from the scan.
hits=$(echo "$files" | grep -v 'scripts/verify_repo.sh' | xargs grep -IlE '/Users/|/home/[a-z]|@gmail\.|api_token|save_account|IBM_QUANTUM_TOKEN|BEGIN [A-Z ]*PRIVATE KEY' 2>/dev/null)
if [ -n "$hits" ]; then echo "FAIL: personal path / credential pattern in:"; echo "$hits"; status=1; else echo "ok: no personal paths or credentials"; fi

for doc in README.md docs/REPRODUCE.md; do
  [ -f "$doc" ] || continue
  for p in $(grep -oE '`[A-Za-z0-9_./-]+\.(py|sh|md|txt|png|json)`' "$doc" | tr -d '`' | sort -u); do
    case "$p" in *\<*|*\**) continue;; esac
    [ -e "$p" ] || { echo "FAIL: $doc names a missing file: $p"; status=1; }
  done
done
[ $status -eq 0 ] && echo "ok: files named in the docs exist"
todo=$(grep -c 'TODO' CITATION.cff 2>/dev/null || true)
[ "${todo:-0}" -gt 0 ] && echo "warning: CITATION.cff still has ${todo} TODO lines (fill them in before publishing)"
exit $status
