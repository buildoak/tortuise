#!/usr/bin/env bash
# DISCLAIMER: This tool is provided for personal, educational, and interoperability
# purposes. It converts SuperSplat's proprietary formats to standard PLY — the same
# data access pattern used by SuperSplat's own MIT-licensed viewer. Users are solely
# responsible for ensuring their use complies with applicable terms of service and
# copyright law. Downloaded content remains the intellectual property of its creator.
# Respect content creators' rights and provide attribution.
# SPDX-License-Identifier: MIT
# supersplat-dl — Download and convert SuperSplat scenes to standard PLY
# Usage: supersplat-dl.sh <url-or-id> [output.ply]
set -euo pipefail

readonly VERSION="1.0.0"
readonly CDN_BASE="https://d28zzqy0iyovbz.cloudfront.net"
readonly COMPRESSED_FILE="scene.compressed.ply"

# ── Helpers ──────────────────────────────────────────────────────────────────

usage() {
    cat <<'USAGE'
supersplat-dl — Download SuperSplat scenes as standard PLY files

USAGE:
    supersplat-dl.sh <url-or-id> [output.ply]

ARGUMENTS:
    <url-or-id>   SuperSplat share URL (https://superspl.at/s?id=...)
                  or just the scene ID (e.g. cf6ac78e)
    [output.ply]  Output file path (default: {id}.ply in current directory)

OPTIONS:
    -h, --help    Show this help message
    -v, --version Show version
    -k, --keep    Keep intermediate compressed PLY file

EXAMPLES:
    supersplat-dl.sh https://superspl.at/s?id=cf6ac78e
    supersplat-dl.sh cf6ac78e output.ply
    supersplat-dl.sh -k cf6ac78e  # keeps the .compressed.ply too
USAGE
    exit 0
}

die() {
    printf "error: %s\n" "$1" >&2
    exit 1
}

info() {
    printf ">> %s\n" "$1"
}

cleanup() {
    if [[ -n "${TMPDIR_CREATED:-}" && -d "${TMPDIR_CREATED}" ]]; then
        rm -rf "${TMPDIR_CREATED}"
    fi
}

trap cleanup EXIT

# ── Argument parsing ─────────────────────────────────────────────────────────

KEEP_COMPRESSED=false
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage ;;
        -v|--version) echo "supersplat-dl ${VERSION}"; exit 0 ;;
        -k|--keep)    KEEP_COMPRESSED=true; shift ;;
        -*)           die "unknown option: $1" ;;
        *)            POSITIONAL+=("$1"); shift ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 1 ]] || die "missing required argument: <url-or-id>. Use -h for help."

INPUT="${POSITIONAL[0]}"
OUTPUT="${POSITIONAL[1]:-}"

# ── Dependency checks ────────────────────────────────────────────────────────

for cmd in python3 curl gunzip; do
    command -v "$cmd" >/dev/null 2>&1 || die "${cmd} is required but not found in PATH"
done

# ── Extract scene ID ─────────────────────────────────────────────────────────

extract_id() {
    local input="$1"

    # Full URL: extract scene ID
    if [[ "$input" =~ ^https?:// ]]; then
        local id
        # Try ?id= or &id= (e.g. superspl.at/s?id=cf6ac78e)
        if [[ "$input" =~ [?\&]id=([a-zA-Z0-9_-]+) ]]; then
            id="${BASH_REMATCH[1]}"
        # Try /scene/{id} path (e.g. superspl.at/scene/3dc55763)
        elif [[ "$input" =~ /scene/([a-zA-Z0-9_-]+) ]]; then
            id="${BASH_REMATCH[1]}"
        else
            die "could not extract scene ID from URL: ${input}"
        fi
        echo "$id"
        return
    fi

    # Bare ID: validate it looks reasonable (hex-ish, 6-64 chars)
    if [[ "$input" =~ ^[a-zA-Z0-9_-]{6,64}$ ]]; then
        echo "$input"
        return
    fi

    die "invalid input: expected a SuperSplat URL or scene ID, got: ${input}"
}

SCENE_ID="$(extract_id "$INPUT")"
info "scene ID: ${SCENE_ID}"

# Default output path
if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${SCENE_ID}.ply"
fi

# ── Create temp directory ────────────────────────────────────────────────────

TMPDIR_CREATED="$(mktemp -d)"
GZIPPED_PATH="${TMPDIR_CREATED}/scene.compressed.ply.gz"
DECOMPRESSED_PATH="${TMPDIR_CREATED}/scene.compressed.ply"
CONVERTED_PATH="${TMPDIR_CREATED}/output.ply"

# ── Resolve script directory ─────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Try compressed PLY first, fall back to SOG ──────────────────────────────

DOWNLOAD_URL="${CDN_BASE}/${SCENE_ID}/v1/${COMPRESSED_FILE}"
info "trying compressed PLY: ${DOWNLOAD_URL}"

CURL_STDERR="${TMPDIR_CREATED}/curl_stderr"
HTTP_CODE="$(curl -sS -w "%{http_code}" -o "${GZIPPED_PATH}" "${DOWNLOAD_URL}" 2>"${CURL_STDERR}")" || HTTP_CODE="000"

if [[ "${HTTP_CODE}" == "200" ]]; then
    # ── Legacy path: compressed PLY ──────────────────────────────────────
    DOWNLOAD_SIZE="$(wc -c < "${GZIPPED_PATH}" | tr -d ' ')"
    info "downloaded: ${DOWNLOAD_SIZE} bytes"

    # Decompress gzip layer if present
    MAGIC="$(od -An -tx1 -N2 "${GZIPPED_PATH}" | tr -d ' \n')"
    if [[ "${MAGIC}" == "1f8b" ]]; then
        info "decompressing gzip layer..."
        if ! gunzip -c "${GZIPPED_PATH}" > "${DECOMPRESSED_PATH}"; then
            die "gzip decompression failed"
        fi
        info "decompressed: $(wc -c < "${DECOMPRESSED_PATH}" | tr -d ' ') bytes"
    else
        mv "${GZIPPED_PATH}" "${DECOMPRESSED_PATH}"
    fi

    # Convert compressed PLY → standard PLY
    CONVERT_PY="${SCRIPT_DIR}/convert.py"
    info "converting compressed PLY → standard PLY..."
    if ! python3 "${CONVERT_PY}" "${DECOMPRESSED_PATH}" "${CONVERTED_PATH}" 2>&1; then
        die "conversion failed. The compressed PLY may be malformed."
    fi
else
    # ── SOG path: newer scenes use WebP-based format ─────────────────────
    info "compressed PLY not available (HTTP ${HTTP_CODE}), trying SOG format..."

    SOG_CONVERT_PY="${SCRIPT_DIR}/sog_convert.py"
    if [[ ! -f "${SOG_CONVERT_PY}" ]]; then
        die "sog_convert.py not found at ${SOG_CONVERT_PY}"
    fi

    if ! python3 "${SOG_CONVERT_PY}" "${SCENE_ID}" "${CONVERTED_PATH}" 2>&1; then
        die "SOG conversion failed. Scene may not exist or format is unsupported."
    fi
fi

if [[ ! -f "${CONVERTED_PATH}" ]]; then
    die "conversion produced no output file"
fi

CONVERTED_SIZE="$(wc -c < "${CONVERTED_PATH}" | tr -d ' ')"
info "converted: ${CONVERTED_SIZE} bytes (standard PLY)"

# ── Validate output ──────────────────────────────────────────────────────────

# Read the PLY text header (everything up to end_header)
# PLY headers are typically <4KB. Read enough to capture it.
HEADER="$(sed -n '1,/^end_header/p' "${CONVERTED_PATH}")"

# Check magic bytes: file must start with "ply"
MAGIC3="$(head -c 3 "${CONVERTED_PATH}")"
if [[ "${MAGIC3}" != "ply" ]]; then
    die "output file does not have a valid PLY magic number"
fi

if ! echo "$HEADER" | grep -q "binary_little_endian"; then
    die "output file is not in binary_little_endian format — tortuise requires this"
fi

# Extract vertex count for reporting
VERTEX_COUNT="$(echo "$HEADER" | grep "element vertex" | head -1 | awk '{print $3}')"
info "vertex count: ${VERTEX_COUNT:-unknown}"

# Check for expected properties
MISSING_PROPS=()
for prop in x y z opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3; do
    if ! echo "$HEADER" | grep -q "property.*${prop}"; then
        MISSING_PROPS+=("$prop")
    fi
done

# Check for color properties (f_dc_* or red/green/blue)
HAS_FDIR=false
HAS_RGB=false
echo "$HEADER" | grep -q "property.*f_dc_0" && HAS_FDIR=true || true
echo "$HEADER" | grep -q "property.*red" && HAS_RGB=true || true

if ! $HAS_FDIR && ! $HAS_RGB; then
    MISSING_PROPS+=("color(f_dc_* or red/green/blue)")
fi

if [[ ${#MISSING_PROPS[@]} -gt 0 ]]; then
    printf "warning: missing properties for tortuise: %s\n" "${MISSING_PROPS[*]}" >&2
fi

# ── Move output ──────────────────────────────────────────────────────────────

# Resolve output path (handle relative paths)
OUTPUT_DIR="$(dirname "${OUTPUT}")"
mkdir -p "${OUTPUT_DIR}"

mv "${CONVERTED_PATH}" "${OUTPUT}"
info "output: ${OUTPUT}"

# Optionally keep compressed file
if $KEEP_COMPRESSED; then
    KEEP_PATH="${OUTPUT_DIR}/${SCENE_ID}.compressed.ply"
    cp "${DECOMPRESSED_PATH}" "${KEEP_PATH}"
    info "kept compressed: ${KEEP_PATH}"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

info "done! ${CONVERTED_SIZE} bytes (standard PLY) — ${VERTEX_COUNT:-?} vertices"
