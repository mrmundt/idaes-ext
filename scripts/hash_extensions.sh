#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") <release> <directory>

Description:
  Computes SHA-256 hashes for all .tar.gz files in <directory>
  and writes them to:

    <directory>/sha256sum_<release>.txt

Arguments:
  <release>    Release label used in the output filename
  <directory>  Directory containing .tar.gz files

Options:
  -h, --help   Show this help message and exit

Example:
  $(basename "$0") v1.2.3 /path/to/artifacts
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 2 ]]; then
    echo "Error: expected 2 arguments." >&2
    usage
    exit 1
fi

RELEASE="$1"
DIR="$2"

OUTFILE="${DIR}/sha256sum_${RELEASE}.txt"

echo "Writing hashes to: ${OUTFILE}"
rm -f "$OUTFILE"
touch "$OUTFILE"

shopt -s nullglob
for f in "$DIR"/*.tar.gz; do
    hash=$(sha256sum "$f" | awk '{print $1}')
    filename=$(basename "$f")
    echo "${hash}  ${filename}" >> "$OUTFILE"
done

echo "Done."

