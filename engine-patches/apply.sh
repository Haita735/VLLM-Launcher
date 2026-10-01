#!/usr/bin/env bash
# Apply the fixes in engine-patches/<engine>/*.diff to an installed engine.
#
#   engine-patches/apply.sh sglang            apply (skips what's already in)
#   engine-patches/apply.sh sglang --check    report only
#   engine-patches/apply.sh sglang --revert   take them out again
#
# The engine's Python is $SGLANG_PYTHON / $VLLM_PYTHON, else the conda env named after the
# engine (the same names the launcher looks for). Each patch is a backport of an upstream fix,
# so a newer engine already has it: a patch that doesn't fit is reported, never forced.
# Reinstalling or upgrading the engine undoes these; run this again afterwards, then restart
# the model so it loads the patched code.
set -euo pipefail

ENGINE=${1:?usage: apply.sh sglang|vllm [--check|--revert]}
MODE=${2:-apply}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

var="$(echo "$ENGINE" | tr a-z A-Z)_PYTHON"
PY=${!var:-}
if [[ -z $PY ]]; then
  for root in ~/miniconda3 ~/anaconda3 ~/miniforge3 ~/mambaforge ~/micromamba /opt/conda /opt/miniconda3 /opt/anaconda3; do
    [[ -x $root/envs/$ENGINE/bin/python ]] && PY=$root/envs/$ENGINE/bin/python && break
  done
fi
[[ -n $PY ]] || { echo "no $ENGINE environment found; set $var" >&2; exit 1; }

SITE=$("$PY" -c "import os, $ENGINE; print(os.path.dirname(os.path.dirname($ENGINE.__file__)))")
VERSION=$("$PY" -c "import $ENGINE; print($ENGINE.__version__)")
echo "$ENGINE $VERSION at $SITE"

shopt -s nullglob
status=0
for diff in "$HERE/$ENGINE"/*.diff; do
  name=$(basename "$diff")
  if patch -d "$SITE" -p1 -F0 -R --dry-run -s < "$diff" > /dev/null 2>&1; then
    if [[ $MODE == --revert ]]; then
      patch -d "$SITE" -p1 -F0 -R -s --no-backup-if-mismatch < "$diff" && echo "  reverted  $name"
    else
      echo "  in        $name"
    fi
  elif patch -d "$SITE" -p1 -F0 --dry-run -s < "$diff" > /dev/null 2>&1; then
    case $MODE in
      apply) patch -d "$SITE" -p1 -F0 -s --no-backup-if-mismatch < "$diff" && echo "  applied   $name" ;;
      --check) echo "  missing   $name"; status=1 ;;
      --revert) echo "  not in    $name" ;;
    esac
  else
    echo "  no fit    $name: this build differs from the one it was made for (a newer one may have the fix already); see the note at the top of it"
    [[ $MODE == --revert ]] || status=1
  fi
done
exit $status
