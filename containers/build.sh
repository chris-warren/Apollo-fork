#!/bin/bash
# Build the training image without root, e.g. on a DRAC login node:
#
#   bash containers/build.sh ~/projects/def-<pi>/<user>/containers/apollo.sif
#
# First tries a normal build of containers/apollo.def (Apptainer's unprivileged
# "fakeroot" mode). If the cluster's Apptainer cannot do that, falls back to building
# a writable sandbox from the same base image, running the definition's %post and
# %test sections in it as you, and packing it into a .sif. Both paths use
# containers/apollo.def as the single description of the environment.
#
# Temporary files go to $APPTAINER_TMPDIR (default: $SCRATCH/apptainer-tmp, or /tmp).
# APOLLO_BUILD_SANDBOX=1 skips straight to the sandbox route.
set -euo pipefail

OUT=${1:?usage: bash containers/build.sh <output.sif>}
HERE=$(cd "$(dirname "$0")" && pwd)
DEF=$HERE/apollo.def
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${SCRATCH:-/tmp}/apptainer-tmp}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${SCRATCH:-$HOME}/.apptainer-cache}
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$OUT")"
command -v apptainer >/dev/null || module load apptainer 2>/dev/null || { echo "apptainer not found (try: module load apptainer)"; exit 1; }

section() {  # print the body of a %section of the definition file
    awk -v s="%$1" '$1 ~ /^%/ {on = ($1 == s); next} on' "$DEF"
}

echo "== building $OUT from $DEF (apptainer $(apptainer --version))"
if [ -z "${APOLLO_BUILD_SANDBOX:-}" ] && apptainer build --fakeroot --ignore-subuid "$OUT" "$DEF"; then
    echo "== built with fakeroot: $OUT"
    exit 0
fi

echo "== building through a sandbox instead"
BASE=$(awk '$1 == "From:" {print $2}' "$DEF")
SANDBOX=$APPTAINER_TMPDIR/apollo-sandbox
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "docker://$BASE"
apptainer exec --writable --no-home --cleanenv "$SANDBOX" bash -euxc "$(section post)"
# A sandbox does not get the definition's %environment; add it before running %test.
mkdir -p "$SANDBOX/.singularity.d/env"
section environment > "$SANDBOX/.singularity.d/env/90-apollo.sh"
apptainer exec --no-home --cleanenv "$SANDBOX" bash -euxc "$(section test)"
apptainer build "$OUT" "$SANDBOX"
rm -rf "$SANDBOX"
echo "== built through sandbox: $OUT"
