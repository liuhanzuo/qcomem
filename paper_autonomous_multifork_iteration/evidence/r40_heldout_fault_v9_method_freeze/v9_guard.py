from __future__ import annotations

import binascii
import hashlib
import io
import json
import os
import re
import stat
import struct
import subprocess
import tarfile
import urllib.parse
import zlib
from pathlib import Path, PurePosixPath


FREEZE_TIMESTAMP = "2026-08-27T00:00:00Z"
TRUST_ROOT_ID = "r40-independent-operator-20260827"
TRUST_ROOT_PUBLIC_KEY_HEX = (
    "a6d6378064dfded46c9810d1687f39835097b86ab459a43a45a308bac69bfa33"
)
TRUST_ROOT_DOCUMENT_SHA256 = (
    "3833a61d8320cc6ac7b510cd51268e73daedb52499a0baf7ee5522b9e9c78440"
)
BINDING_DOMAIN = b"forkaudit-v9-operator-binding-payload-v1\x00"
TERM_IDS = tuple(f"V9F{x:02d}" for x in range(1, 9))
HASH_KEYS = (
    "archive_sha256",
    "source_ledger_sha256",
    "snapshot_sha256",
    "snapshot_inventory_sha256",
    "runner_manifest_sha256",
    "runner_inventory_sha256",
)

ENV_LITERAL_POLICY = {
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
}
ENV_PATH_KEYS = ("FORKAUDIT_CONFIG_PATH",)

SPAWN_PROVENANCE_KEYS = (
    "binding_sha256",
    "consumption_sha256",
    "execution_contract_sha256",
    "probe_report_sha256",
    "runtime_expectation_sha256",
    "spawned_specs_sha256",
)


class Reject(ValueError):
    pass


def need(value: object, message: str) -> None:
    if not value:
        raise Reject(message)


def exact_dict(value: object, keys: tuple[str, ...], message: str) -> dict:
    need(type(value) is dict and set(value) == set(keys), message)
    return value  # type: ignore[return-value]


def text(value: object, message: str) -> str:
    need(type(value) is str and bool(value), message)
    return value  # type: ignore[return-value]


def integer(value: object, message: str) -> int:
    need(type(value) is int, message)
    return value  # type: ignore[return-value]


def sha256_hex(value: object, message: str) -> str:
    need(
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        message,
    )
    return value  # type: ignore[return-value]


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _no_constant(value: str) -> None:
    raise Reject(f"non-JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        need(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def canonical_load(raw: bytes, message: str) -> object:
    need(type(raw) is bytes, message)
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_no_constant,
        )
    except Reject:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Reject(message) from exc
    need(raw == canonical_bytes(value), f"noncanonical {message}")
    return value


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: os.PathLike[str] | str) -> tuple[bytes, os.stat_result]:
    p = Path(path)
    try:
        before = os.lstat(p)
    except OSError as exc:
        raise Reject(f"unreadable file: {p}") from exc
    need(stat.S_ISREG(before.st_mode), f"not regular: {p}")
    need(before.st_nlink == 1, f"not unique: {p}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(p, flags)
    except OSError as exc:
        raise Reject(f"cannot open regular file: {p}") from exc
    try:
        opened = os.fstat(fd)
        need(
            stat.S_ISREG(opened.st_mode)
            and opened.st_nlink == 1
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"file changed before read: {p}",
        )
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        need(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            f"file changed during read: {p}",
        )
        return b"".join(chunks), after
    finally:
        os.close(fd)


def digest_file(path: os.PathLike[str] | str) -> str:
    return digest_bytes(_read_regular(path)[0])


def _member_name(name: object) -> str:
    value = text(name, "path")
    need("\\" not in value and not value.startswith("/"), "noncanonical path")
    parts = PurePosixPath(value).parts
    need(parts and all(x not in ("", ".", "..") for x in parts), "traversal path")
    need(PurePosixPath(value).as_posix() == value, "noncanonical path")
    return value


def _scan_closed_tree(root: os.PathLike[str] | str) -> list[tuple[str, bytes]]:
    root_path = Path(root)
    try:
        root_stat = os.lstat(root_path)
    except OSError as exc:
        raise Reject("tree root absent") from exc
    need(stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode), "tree root")
    files: list[tuple[str, bytes]] = []
    inodes: set[tuple[int, int]] = set()

    def visit(directory: Path, prefix: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise Reject("tree scan") from exc
        for entry in entries:
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            _member_name(rel)
            try:
                item_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise Reject("tree stat") from exc
            need(not stat.S_ISLNK(item_stat.st_mode), "tree symlink")
            if stat.S_ISDIR(item_stat.st_mode):
                visit(Path(entry.path), rel)
                continue
            need(stat.S_ISREG(item_stat.st_mode) and item_stat.st_nlink == 1, "tree regular unique")
            key = (item_stat.st_dev, item_stat.st_ino)
            need(key not in inodes, "duplicate tree inode")
            inodes.add(key)
            raw, opened = _read_regular(entry.path)
            need((opened.st_dev, opened.st_ino) == key, "tree race")
            files.append((rel, raw))

    visit(root_path, "")
    need(bool(files), "empty closed tree")
    return files


def _inventory(files: list[tuple[str, bytes]], schema: str) -> tuple[list[dict], str]:
    rows = [
        {"path": name, "sha256": digest_bytes(raw), "size": len(raw)}
        for name, raw in files
    ]
    rows.sort(key=lambda row: row["path"])
    inventory = {"schema_version": schema, "files": rows}
    return rows, digest_bytes(canonical_bytes(inventory))


def snapshot_commitments(root: os.PathLike[str] | str) -> tuple[str, str, list[dict]]:
    files = _scan_closed_tree(root)
    rows, inventory_hash = _inventory(files, "forkaudit-v9-snapshot-inventory-v1")
    commitment = hashlib.sha256(b"forkaudit-v9-snapshot-bytes-v1\x00")
    for name, raw in files:
        name_raw = name.encode("utf-8")
        commitment.update(struct.pack(">Q", len(name_raw)))
        commitment.update(name_raw)
        commitment.update(struct.pack(">Q", len(raw)))
        commitment.update(raw)
    return commitment.hexdigest(), inventory_hash, rows


def _manifest_rows(manifest: object) -> list[dict]:
    need(type(manifest) is list and bool(manifest), "runner manifest")
    rows: list[dict] = []
    previous = None
    for item in manifest:  # type: ignore[union-attr]
        row = exact_dict(item, ("path", "sha256", "size"), "runner row")
        path = _member_name(row["path"])
        sha256_hex(row["sha256"], "runner hash")
        size = integer(row["size"], "runner size")
        need(size >= 0 and (previous is None or previous < path), "runner row order/duplicate")
        previous = path
        rows.append({"path": path, "sha256": row["sha256"], "size": size})
    return rows


def runner_commitments(
    root: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    expected_manifest_sha256: object | None = None,
    expected_inventory_sha256: object | None = None,
) -> tuple[str, str, list[dict]]:
    """Validate canonical manifest bytes and independently rescan the closed tree.

    The manifest is always loaded from a unique regular file.  No caller-supplied
    Python object can stand in for the signed bytes.
    """
    need(isinstance(manifest_path, (str, os.PathLike)), "runner manifest must be a file path")
    manifest_raw, _ = _read_regular(manifest_path)
    manifest_hash = digest_bytes(manifest_raw)
    if expected_manifest_sha256 is not None:
        need(
            manifest_hash
            == sha256_hex(expected_manifest_sha256, "runner manifest anchor"),
            "runner manifest digest",
        )
    expected = _manifest_rows(canonical_load(manifest_raw, "runner manifest"))
    files = _scan_closed_tree(root)
    actual, inventory_hash = _inventory(files, "forkaudit-v9-runner-inventory-v1")
    need(actual == expected, "runner inventory mismatch")
    if expected_inventory_sha256 is not None:
        need(
            inventory_hash
            == sha256_hex(expected_inventory_sha256, "runner inventory anchor"),
            "runner inventory digest",
        )
    return manifest_hash, inventory_hash, actual


def validate_runner_tree(
    root: os.PathLike[str] | str,
    manifest_path: os.PathLike[str] | str,
    expected_manifest_sha256: object,
    expected_inventory_sha256: object,
) -> tuple[str, list[dict]]:
    _, inventory_hash, actual = runner_commitments(
        root,
        manifest_path,
        expected_manifest_sha256,
        expected_inventory_sha256,
    )
    return inventory_hash, actual


def validate_hash_map(value: object, *, allow_null: bool = False) -> dict:
    mapping = exact_dict(value, HASH_KEYS, "hash map schema")
    out = {}
    for key in HASH_KEYS:
        if allow_null and mapping[key] is None:
            out[key] = None
        else:
            out[key] = sha256_hex(mapping[key], key)
    return out


def measure_hashes(
    archive_path: os.PathLike[str] | str,
    source_ledger_path: os.PathLike[str] | str,
    snapshot_root: os.PathLike[str] | str,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
) -> dict:
    snapshot_hash, snapshot_inventory_hash, _ = snapshot_commitments(snapshot_root)
    runner_manifest_hash, runner_inventory_hash, _ = runner_commitments(
        runner_root, runner_manifest_path
    )
    return {
        "archive_sha256": digest_file(archive_path),
        "source_ledger_sha256": digest_file(source_ledger_path),
        "snapshot_sha256": snapshot_hash,
        "snapshot_inventory_sha256": snapshot_inventory_hash,
        "runner_manifest_sha256": runner_manifest_hash,
        "runner_inventory_sha256": runner_inventory_hash,
    }


def canonical_tar(entries: list[tuple[str, bytes]]) -> bytes:
    checked = []
    seen = set()
    for name, raw in entries:
        canonical = _member_name(name)
        need(canonical not in seen and type(raw) is bytes, "duplicate/archive entry")
        seen.add(canonical)
        checked.append((canonical, raw))
    checked.sort(key=lambda pair: pair[0])
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw in checked:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def deterministic_gzip(raw: bytes) -> bytes:
    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    body = compressor.compress(raw) + compressor.flush()
    header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    trailer = struct.pack("<II", binascii.crc32(raw) & 0xFFFFFFFF, len(raw) & 0xFFFFFFFF)
    return header + body + trailer


def strict_gzip(raw: bytes) -> bytes:
    need(len(raw) >= 18, "short gzip")
    need(raw[:10] == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff", "gzip header")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        plain = decompressor.decompress(raw[10:]) + decompressor.flush()
    except zlib.error as exc:
        raise Reject("gzip deflate") from exc
    need(decompressor.eof and len(decompressor.unused_data) == 8, "multiple gzip streams/trailing bytes")
    crc, size = struct.unpack("<II", decompressor.unused_data)
    need(crc == (binascii.crc32(plain) & 0xFFFFFFFF), "gzip crc")
    need(size == (len(plain) & 0xFFFFFFFF), "gzip size")
    need(raw == deterministic_gzip(plain), "noncanonical gzip encoding")
    return plain


def verify_archive(
    path: os.PathLike[str] | str,
    expected_archive_sha256: object,
    expected_ledger_sha256: object,
) -> dict:
    archive_raw, _ = _read_regular(path)
    need(
        digest_bytes(archive_raw) == sha256_hex(expected_archive_sha256, "archive anchor"),
        "archive digest",
    )
    tar_raw = strict_gzip(archive_raw)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(tar_raw), mode="r:")
        members = archive.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise Reject("tar parse") from exc
    try:
        need(bool(members), "empty archive")
        names = []
        entries = []
        for member in members:
            name = _member_name(member.name)
            need(name not in names, "duplicate archive name")
            names.append(name)
            need(
                member.isfile()
                and member.mode == 0o444
                and member.uid == member.gid == member.mtime == 0
                and member.uname == member.gname == ""
                and member.linkname == ""
                and member.devmajor == member.devminor == 0
                and not member.pax_headers,
                "noncanonical tar metadata",
            )
            extracted = archive.extractfile(member)
            need(extracted is not None, "archive member")
            entries.append((name, extracted.read()))
        need(names == sorted(names), "archive member order")
        need(tar_raw == canonical_tar(entries), "noncanonical tar bytes")
    finally:
        archive.close()
    by_name = dict(entries)
    need("source-ledger.json" in by_name, "missing ledger")
    ledger_raw = by_name["source-ledger.json"]
    need(
        digest_bytes(ledger_raw) == sha256_hex(expected_ledger_sha256, "ledger anchor"),
        "ledger digest",
    )
    ledger = canonical_load(ledger_raw, "source ledger")
    ledger_dict = exact_dict(ledger, ("schema_version", "freeze_timestamp", "files"), "ledger schema")
    need(ledger_dict["schema_version"] == "forkaudit-v9-source-ledger-v1", "ledger version")
    need(ledger_dict["freeze_timestamp"] == FREEZE_TIMESTAMP, "ledger timestamp")
    rows = _manifest_rows(ledger_dict["files"])
    need({row["path"] for row in rows} == set(by_name) - {"source-ledger.json"}, "exact ledger inventory")
    for row in rows:
        member_raw = by_name[row["path"]]
        need(
            row["size"] == len(member_raw) and row["sha256"] == digest_bytes(member_raw),
            "ledger member mismatch",
        )
    return {"ledger_bytes": ledger_raw, "members": len(entries), "files": rows}


# Minimal strict Ed25519 verification (RFC 8032 equation, canonical encodings).
_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_I = pow(2, (_P - 1) // 4, _P)
_IDENTITY = (0, 1)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    x = pow(xx, (_P + 3) // 8, _P)
    if (x * x - xx) % _P:
        x = x * _I % _P
    need((x * x - xx) % _P == 0, "ed25519 point")
    return x


def _decode_point(raw: bytes) -> tuple[int, int]:
    need(len(raw) == 32, "ed25519 point length")
    encoded = int.from_bytes(raw, "little")
    y = encoded & ((1 << 255) - 1)
    sign_bit = encoded >> 255
    need(y < _P, "noncanonical ed25519 point")
    x = _xrecover(y)
    if (x & 1) != sign_bit:
        x = _P - x
    need(not (x == 0 and sign_bit == 1), "noncanonical ed25519 sign")
    need((-x * x + y * y - 1 - _D * x * x * y * y) % _P == 0, "ed25519 curve")
    return x, y


def _point_add(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = left
    x2, y2 = right
    product = _D * x1 * x2 * y1 * y2 % _P
    x3 = (x1 * y2 + x2 * y1) * pow(1 + product, _P - 2, _P) % _P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - product, _P - 2, _P) % _P
    return x3, y3


def _scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = _IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        scalar >>= 1
    return result


_BASE_Y = 4 * pow(5, _P - 2, _P) % _P
_BASE_X = _xrecover(_BASE_Y)
if _BASE_X & 1:
    _BASE_X = _P - _BASE_X
_BASE = (_BASE_X, _BASE_Y)


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    try:
        need(len(public_key) == 32 and len(signature) == 64, "ed25519 lengths")
        public_point = _decode_point(public_key)
        r_point = _decode_point(signature[:32])
        scalar = int.from_bytes(signature[32:], "little")
        need(scalar < _L, "noncanonical ed25519 scalar")
        # Pinned keys and signatures must not be small-order points.
        need(_scalar_mult(public_point, 8) != _IDENTITY, "small-order public key")
        need(_scalar_mult(r_point, 8) != _IDENTITY, "small-order signature point")
        challenge = int.from_bytes(
            hashlib.sha512(signature[:32] + public_key + message).digest(), "little"
        ) % _L
        return _scalar_mult(_BASE, scalar) == _point_add(
            r_point, _scalar_mult(public_point, challenge)
        )
    except (Reject, ValueError, OverflowError):
        return False


def _compiled_trust_root() -> None:
    path = Path(__file__).resolve().parent / "OPERATOR_TRUST_ROOT.json"
    raw, _ = _read_regular(path)
    need(digest_bytes(raw) == TRUST_ROOT_DOCUMENT_SHA256, "trust-root document pin")
    root = canonical_load(raw, "trust root")
    root_dict = exact_dict(root, ("algorithm", "key_id", "public_key_hex", "schema_version"), "trust root schema")
    need(root_dict["schema_version"] == "forkaudit-v9-operator-trust-root-v1", "trust root version")
    need(root_dict["algorithm"] == "ed25519", "trust root algorithm")
    need(root_dict["key_id"] == TRUST_ROOT_ID, "trust root id")
    need(root_dict["public_key_hex"] == TRUST_ROOT_PUBLIC_KEY_HEX, "trust root key")


def _published_uri(value: object) -> str:
    uri = text(value, "publication URI")
    parsed = urllib.parse.urlsplit(uri)
    need(
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
        and parsed.path.startswith("/")
        and ".." not in PurePosixPath(parsed.path).parts
        and parsed.geturl() == uri,
        "canonical public HTTPS URI",
    )
    return uri


def verify_operator_binding(
    raw_binding: bytes,
    archive_path: os.PathLike[str] | str,
    source_ledger_path: os.PathLike[str] | str,
    snapshot_root: os.PathLike[str] | str,
    execution_contract_path: os.PathLike[str] | str,
    runtime_expectation_path: os.PathLike[str] | str,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
    terminal_root: os.PathLike[str] | str,
    attempt: object,
    run_nonce: object,
) -> dict:
    _compiled_trust_root()
    binding = canonical_load(raw_binding, "signed operator binding")
    outer = exact_dict(binding, ("payload", "schema_version", "signature"), "binding schema")
    need(outer["schema_version"] == "forkaudit-v9-signed-operator-binding-v1", "binding version")
    payload = exact_dict(
        outer["payload"],
        (
            "approved_archive_sha256",
            "approved_attempt",
            "approved_execution_contract_sha256",
            "approved_run_nonce",
            "approved_runner_inventory_sha256",
            "approved_runner_manifest_sha256",
            "approved_runtime_expectation_sha256",
            "approved_snapshot_inventory_sha256",
            "approved_snapshot_sha256",
            "approved_source_ledger_sha256",
            "approved_terminal_root",
            "operator_id",
            "published_uri",
            "schema_version",
            "trust_root_id",
        ),
        "binding payload",
    )
    signature = exact_dict(outer["signature"], ("algorithm", "key_id", "signature_hex"), "binding signature")
    need(payload["schema_version"] == "forkaudit-v9-operator-binding-payload-v1", "payload version")
    need(payload["trust_root_id"] == TRUST_ROOT_ID, "payload trust root")
    text(payload["operator_id"], "operator id")
    _published_uri(payload["published_uri"])
    for key in (
        "approved_archive_sha256",
        "approved_execution_contract_sha256",
        "approved_runner_inventory_sha256",
        "approved_runner_manifest_sha256",
        "approved_runtime_expectation_sha256",
        "approved_snapshot_inventory_sha256",
        "approved_snapshot_sha256",
        "approved_source_ledger_sha256",
    ):
        sha256_hex(payload[key], key)
    signed_attempt = integer(payload["approved_attempt"], "approved attempt")
    need(signed_attempt > 0 and type(attempt) is int and attempt == signed_attempt, "signed attempt")
    signed_nonce = sha256_hex(payload["approved_run_nonce"], "approved run nonce")
    need(sha256_hex(run_nonce, "run nonce") == signed_nonce, "signed run nonce")
    signed_terminal = text(payload["approved_terminal_root"], "approved terminal root")
    need(Path(signed_terminal).is_absolute(), "terminal root must be absolute")
    try:
        signed_resolved = Path(signed_terminal).resolve(strict=True)
        actual_resolved = Path(terminal_root).resolve(strict=True)
    except OSError as exc:
        raise Reject("terminal root resolution") from exc
    need(str(signed_resolved) == signed_terminal, "canonical signed terminal root")
    signed_stat = os.lstat(signed_resolved)
    actual_stat = os.lstat(actual_resolved)
    need(
        stat.S_ISDIR(signed_stat.st_mode)
        and not stat.S_ISLNK(signed_stat.st_mode)
        and (signed_stat.st_dev, signed_stat.st_ino)
        == (actual_stat.st_dev, actual_stat.st_ino),
        "signed terminal root identity",
    )
    need(signature["algorithm"] == "ed25519" and signature["key_id"] == TRUST_ROOT_ID, "signature root")
    signature_hex = text(signature["signature_hex"], "signature")
    need(len(signature_hex) == 128 and all(c in "0123456789abcdef" for c in signature_hex), "signature encoding")
    message = BINDING_DOMAIN + canonical_bytes(payload)
    need(
        ed25519_verify(
            bytes.fromhex(TRUST_ROOT_PUBLIC_KEY_HEX), message, bytes.fromhex(signature_hex)
        ),
        "operator signature",
    )

    archive_result = verify_archive(
        archive_path,
        payload["approved_archive_sha256"],
        payload["approved_source_ledger_sha256"],
    )
    ledger_raw, _ = _read_regular(source_ledger_path)
    need(ledger_raw == archive_result["ledger_bytes"], "external/embedded ledger mismatch")
    snapshot_hash, snapshot_inventory_hash, _ = snapshot_commitments(snapshot_root)
    need(snapshot_hash == payload["approved_snapshot_sha256"], "snapshot byte commitment")
    need(
        snapshot_inventory_hash == payload["approved_snapshot_inventory_sha256"],
        "snapshot inventory commitment",
    )
    manifest_sha, inventory_sha, _ = runner_commitments(
        runner_root,
        runner_manifest_path,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    need(
        manifest_sha == payload["approved_runner_manifest_sha256"]
        and inventory_sha == payload["approved_runner_inventory_sha256"],
        "signed runner commitments",
    )
    contract_raw, _ = _read_regular(execution_contract_path)
    expectation_raw, _ = _read_regular(runtime_expectation_path)
    validate_runtime_expectation(
        expectation_raw,
        payload["approved_runtime_expectation_sha256"],
        runner_root,
        runner_manifest_path,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    validate_execution_contract(
        contract_raw,
        payload["approved_execution_contract_sha256"],
        runtime_expectation_path,
        runner_root,
        runner_manifest_path,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
        payload["approved_runtime_expectation_sha256"],
    )
    return payload


def designer_attestation(
    value: object, expected_snapshot_sha256: object, expected_inventory_sha256: object
) -> None:
    attestation = exact_dict(
        value,
        (
            "inputs_limited_to_snapshot",
            "no_prior_faults_seen",
            "no_private_source_seen",
            "snapshot_inventory_sha256",
            "snapshot_sha256",
        ),
        "designer attestation",
    )
    need(
        sha256_hex(attestation["snapshot_sha256"], "snapshot")
        == sha256_hex(expected_snapshot_sha256, "expected snapshot"),
        "snapshot attestation mismatch",
    )
    need(
        sha256_hex(attestation["snapshot_inventory_sha256"], "snapshot inventory")
        == sha256_hex(expected_inventory_sha256, "expected snapshot inventory"),
        "snapshot inventory attestation mismatch",
    )
    for key in ("inputs_limited_to_snapshot", "no_prior_faults_seen", "no_private_source_seen"):
        need(attestation[key] is True, key)


def _safe_cwd(root: Path, value: object) -> tuple[str, Path]:
    cwd = text(value, "cwd")
    need("\\" not in cwd and not Path(cwd).is_absolute(), "cwd must be relative")
    need(cwd == "." or _member_name(cwd) == cwd, "canonical cwd")
    resolved = root if cwd == "." else root / cwd
    current = root
    if cwd != ".":
        for part in PurePosixPath(cwd).parts:
            current = current / part
            item = os.lstat(current)
            need(stat.S_ISDIR(item.st_mode) and not stat.S_ISLNK(item.st_mode), "cwd tree")
    need(resolved.is_dir(), "cwd absent")
    return cwd, resolved


def _bound_manifest_path(
    root: Path,
    cwd_path: Path,
    spelling_value: object,
    manifest_path_value: object,
    by_path: dict[str, dict],
) -> str:
    spelling = text(spelling_value, "path spelling")
    need("\x00" not in spelling and "\\" not in spelling and not spelling.startswith("~"), "path spelling")
    manifest_path = _member_name(manifest_path_value)
    need(manifest_path in by_path, "path not in signed runner inventory")
    candidate = Path(spelling)
    lexical = candidate if candidate.is_absolute() else cwd_path / candidate
    try:
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise Reject("path escapes runner root") from exc
    need(relative == manifest_path, "path binding target")
    target_stat = os.lstat(resolved)
    need(stat.S_ISREG(target_stat.st_mode) and target_stat.st_nlink == 1, "bound path regular unique")
    need(
        digest_file(resolved) == by_path[manifest_path]["sha256"]
        and target_stat.st_size == by_path[manifest_path]["size"],
        "bound path content",
    )
    return spelling


def reject_known_shell_executable(path: os.PathLike[str] | str) -> None:
    """Reject a shell masquerading as the expected Python before any subprocess.

    The external operator still signs the exact interpreter bytes.  This local
    check closes the concrete `/bin/sh` transplantation counterexample without
    claiming generic hostile-binary recognition.
    """
    candidate_raw, _ = _read_regular(path)
    first_line = candidate_raw.split(b"\n", 1)[0][:512].lower()
    need(
        not (
            first_line.startswith(b"#!")
            and any(token in first_line for token in (b"/sh", b"bash", b"zsh", b"dash", b"ksh"))
        ),
        "shell script cannot be Python",
    )
    def reference_bytes(reference: Path) -> bytes:
        resolved = reference.resolve(strict=True)
        before = os.lstat(resolved)
        need(stat.S_ISREG(before.st_mode), "shell reference regular")
        fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            need(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                "shell reference race",
            )
            chunks = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    for shell in (
        "/bin/sh",
        "/bin/bash",
        "/bin/zsh",
        "/bin/dash",
        "/bin/ksh",
        "/usr/bin/sh",
        "/usr/bin/bash",
        "/usr/bin/zsh",
    ):
        shell_path = Path(shell)
        if not shell_path.is_file():
            continue
        try:
            need(candidate_raw != reference_bytes(shell_path), "shell binary cannot be Python")
        except Reject as exc:
            if str(exc) == "shell binary cannot be Python":
                raise
            # An unreadable platform reference is irrelevant to the candidate's
            # independent unique-regular-file validation.
            continue


def _validate_argv_schema(
    argv_value: object,
    schema_value: object,
    root: Path,
    cwd_path: Path,
    by_path: dict[str, dict],
) -> list[str]:
    need(type(argv_value) is list and bool(argv_value), "argv")
    argv = []
    for item in argv_value:  # type: ignore[union-attr]
        token = text(item, "argv item")
        need("\x00" not in token, "argv NUL")
        argv.append(token)
    need(type(schema_value) is list and len(schema_value) == len(argv), "argv schema cardinality")
    for expected_index, item in enumerate(schema_value):  # type: ignore[union-attr]
        need(type(item) is dict and "kind" in item, "argv typed row")
        kind = item["kind"]
        if kind == "literal":
            row = exact_dict(item, ("index", "kind", "literal"), "argv literal row")
            literal = text(row["literal"], "argv literal")
            need(argv[expected_index] == literal and "\x00" not in literal, "argv literal mismatch")
        elif kind == "option":
            row = exact_dict(item, ("index", "kind", "option"), "argv option row")
            option = text(row["option"], "argv option")
            need(option.startswith("-") and "=" not in option and argv[expected_index] == option, "argv option mismatch")
        elif kind == "path":
            row = exact_dict(
                item,
                ("index", "kind", "manifest_path", "spelling"),
                "argv path row",
            )
            spelling = _bound_manifest_path(
                root, cwd_path, row["spelling"], row["manifest_path"], by_path
            )
            need(argv[expected_index] == spelling, "argv path spelling mismatch")
        elif kind == "option-path":
            row = exact_dict(
                item,
                ("index", "kind", "manifest_path", "option", "spelling"),
                "argv option-path row",
            )
            option = text(row["option"], "argv path option")
            need(option.startswith("-") and "=" not in option, "argv path option")
            spelling = _bound_manifest_path(
                root, cwd_path, row["spelling"], row["manifest_path"], by_path
            )
            need(argv[expected_index] == f"{option}={spelling}", "argv option-path mismatch")
        else:
            raise Reject("unknown argv kind")
        need(integer(row["index"], "argv typed index") == expected_index, "argv typed index/order")
    need(schema_value[0]["kind"] == "path", "argv[0] must be typed executable path")
    return argv


def _validate_env_schema(
    env_value: object,
    schema_value: object,
    root: Path,
    cwd_path: Path,
    by_path: dict[str, dict],
) -> dict[str, str]:
    required_keys = set(ENV_LITERAL_POLICY) | {"CUDA_VISIBLE_DEVICES"} | set(ENV_PATH_KEYS)
    env = exact_dict(env_value, tuple(sorted(required_keys)), "strict environment allowlist")
    for key, value in env.items():
        need(type(value) is str and "\x00" not in key + value, "env item")
    need(type(schema_value) is list and len(schema_value) == len(env), "env schema cardinality")
    need([item.get("key") if type(item) is dict else None for item in schema_value] == sorted(env), "env schema order")
    for item in schema_value:  # type: ignore[union-attr]
        need(type(item) is dict and "kind" in item, "env typed row")
        key = text(item.get("key"), "env key")
        kind = item["kind"]
        if key in ENV_LITERAL_POLICY:
            row = exact_dict(item, ("key", "kind", "literal"), "env literal row")
            need(kind == "literal" and row["literal"] == ENV_LITERAL_POLICY[key], "env literal policy")
            need(env[key] == row["literal"], "env literal mismatch")
        elif key == "CUDA_VISIBLE_DEVICES":
            row = exact_dict(item, ("key", "kind", "visibility"), "env UUID row")
            need(kind == "uuid-list" and row["visibility"] == env[key], "env visibility mismatch")
            canonical_visibility(env[key])
        elif key in ENV_PATH_KEYS:
            row = exact_dict(
                item,
                ("key", "kind", "manifest_path", "spelling"),
                "env path row",
            )
            need(kind == "path", "env path kind")
            spelling = _bound_manifest_path(
                root, cwd_path, row["spelling"], row["manifest_path"], by_path
            )
            need(env[key] == spelling, "env path mismatch")
        else:
            raise Reject("environment key outside allowlist")
    return env  # type: ignore[return-value]


def validate_execution_contract(
    raw_contract: bytes,
    expected_contract_sha256: object,
    runtime_expectation_path: os.PathLike[str] | str,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
    expected_runner_manifest_sha256: object,
    expected_runner_inventory_sha256: object,
    expected_runtime_expectation_sha256: object,
) -> dict:
    need(digest_bytes(raw_contract) == sha256_hex(expected_contract_sha256, "contract anchor"), "contract digest")
    contract = canonical_load(raw_contract, "execution contract")
    value = exact_dict(
        contract,
        (
            "runner_inventory_sha256",
            "runner_manifest_sha256",
            "runtime_expectation_sha256",
            "schema_version",
            "timeout_seconds",
            "workers",
        ),
        "execution contract schema",
    )
    need(value["schema_version"] == "forkaudit-v9-execution-contract-v1", "contract version")
    manifest_anchor = sha256_hex(expected_runner_manifest_sha256, "expected runner manifest")
    inventory_anchor = sha256_hex(expected_runner_inventory_sha256, "expected runner inventory")
    expectation_anchor = sha256_hex(expected_runtime_expectation_sha256, "expected runtime expectation")
    need(value["runner_manifest_sha256"] == manifest_anchor, "contract runner manifest anchor")
    need(value["runner_inventory_sha256"] == inventory_anchor, "contract runner inventory anchor")
    need(value["runtime_expectation_sha256"] == expectation_anchor, "contract runtime expectation anchor")
    timeout = integer(value["timeout_seconds"], "signed worker timeout")
    need(0 < timeout <= 3600, "signed worker timeout range")

    root = Path(runner_root)
    root_stat = os.lstat(root)
    need(stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode), "runner root")
    _, actual_rows = validate_runner_tree(
        root,
        runner_manifest_path,
        manifest_anchor,
        inventory_anchor,
    )
    by_path = {row["path"]: row for row in actual_rows}
    expectation_raw, _ = _read_regular(runtime_expectation_path)
    expectation = validate_runtime_expectation(
        expectation_raw,
        expectation_anchor,
        root,
        runner_manifest_path,
        manifest_anchor,
        inventory_anchor,
    )
    expected_python = str(
        (root / expectation["python"]["manifest_path"]).resolve(strict=True)
    )
    reject_known_shell_executable(expected_python)
    expected_cwd = expectation["cwd"]
    expected_visibility = expectation["device"]["visibility"]
    expected_index = expectation["device"]["index"]
    expected_uuid = expectation["device"]["physical_uuid"]

    workers = value["workers"]
    need(type(workers) is list and len(workers) == len(TERM_IDS), "exact eight workers")
    parsed_workers = []
    for expected_fault_id, worker in zip(TERM_IDS, workers):
        row = exact_dict(
            worker,
            ("argv", "argv_schema", "cwd", "env", "env_schema", "fault_id"),
            "worker contract schema",
        )
        need(row["fault_id"] == expected_fault_id, "worker fault id/order")
        cwd, cwd_path = _safe_cwd(root, row["cwd"])
        argv = _validate_argv_schema(
            row["argv"], row["argv_schema"], root, cwd_path, by_path
        )
        env = _validate_env_schema(
            row["env"], row["env_schema"], root, cwd_path, by_path
        )
        need(argv[0] == expected_python, "worker Python/runtime expectation cross-link")
        need(cwd == expected_cwd, "worker cwd/runtime expectation cross-link")
        need(
            env["CUDA_VISIBLE_DEVICES"] == expected_visibility,
            "worker visibility/runtime expectation cross-link",
        )
        visible = canonical_visibility(env["CUDA_VISIBLE_DEVICES"])
        need(
            visible[expected_index] == expected_uuid,
            "worker UUID/runtime expectation cross-link",
        )
        parsed_workers.append(row)
    need([row["fault_id"] for row in parsed_workers] == list(TERM_IDS), "closed worker id set")
    return value


def derive_spawned_specs(contract: object, runner_root: os.PathLike[str] | str) -> dict:
    """Derive the only actual Popen arguments from an already validated contract."""
    value = exact_dict(
        contract,
        (
            "runner_inventory_sha256",
            "runner_manifest_sha256",
            "runtime_expectation_sha256",
            "schema_version",
            "timeout_seconds",
            "workers",
        ),
        "validated execution contract",
    )
    need(value["schema_version"] == "forkaudit-v9-execution-contract-v1", "contract version")
    root = Path(runner_root).resolve(strict=True)
    contract_workers = value["workers"]
    need(type(contract_workers) is list and len(contract_workers) == len(TERM_IDS), "derived worker cardinality")
    workers = []
    for expected_fault_id, row in zip(TERM_IDS, contract_workers):
        worker = exact_dict(
            row,
            ("argv", "argv_schema", "cwd", "env", "env_schema", "fault_id"),
            "validated worker contract",
        )
        need(worker["fault_id"] == expected_fault_id, "derived worker order")
        _, cwd_path = _safe_cwd(root, worker["cwd"])
        workers.append(
            {
                "argv": list(worker["argv"]),
                "argv_schema": json.loads(json.dumps(worker["argv_schema"])),
                "cwd": str(cwd_path.resolve(strict=True)),
                "cwd_contract": worker["cwd"],
                "env": dict(worker["env"]),
                "env_schema": json.loads(json.dumps(worker["env_schema"])),
                "fault_id": expected_fault_id,
            }
        )
    need(len(workers) == len(TERM_IDS), "derived exact worker set")
    return {
        "schema_version": "forkaudit-v9-spawned-specs-v1",
        "workers": workers,
    }


_UUID_RE = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def canonical_uuid(value: object) -> str:
    uuid = text(value, "GPU UUID")
    need(_UUID_RE.fullmatch(uuid) is not None, "canonical GPU UUID")
    return uuid


def canonical_visibility(value: object) -> tuple[str, ...]:
    visibility = text(value, "CUDA visibility")
    parts = visibility.split(",")
    need(bool(parts) and all(canonical_uuid(part) == part for part in parts), "canonical visibility")
    need(len(set(parts)) == len(parts), "duplicate visibility UUID")
    need(",".join(parts) == visibility, "visibility spelling")
    return tuple(parts)


def validate_runtime_expectation(
    raw_expectation: bytes,
    expected_expectation_sha256: object,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
    expected_runner_manifest_sha256: object,
    expected_runner_inventory_sha256: object,
) -> dict:
    need(
        digest_bytes(raw_expectation)
        == sha256_hex(expected_expectation_sha256, "runtime expectation anchor"),
        "runtime expectation digest",
    )
    expectation = canonical_load(raw_expectation, "runtime expectation")
    value = exact_dict(
        expectation,
        (
            "cwd",
            "device",
            "probe",
            "python",
            "runner_inventory_sha256",
            "runner_manifest_sha256",
            "runtime_closure",
            "schema_version",
            "torch",
        ),
        "runtime expectation schema",
    )
    need(
        value["schema_version"] == "forkaudit-v9-runtime-expectation-v1",
        "runtime expectation version",
    )
    manifest_anchor = sha256_hex(expected_runner_manifest_sha256, "expected runner manifest")
    inventory_anchor = sha256_hex(expected_runner_inventory_sha256, "expected runner inventory")
    need(value["runner_manifest_sha256"] == manifest_anchor, "expectation runner manifest anchor")
    need(value["runner_inventory_sha256"] == inventory_anchor, "expectation runner inventory anchor")
    root = Path(runner_root)
    _, rows = validate_runner_tree(
        root,
        runner_manifest_path,
        manifest_anchor,
        inventory_anchor,
    )
    by_path = {row["path"]: row for row in rows}
    _safe_cwd(root, value["cwd"])

    python = exact_dict(
        value["python"],
        ("cache_tag", "implementation", "manifest_path", "sha256", "version"),
        "python expectation",
    )
    need(python["implementation"] == "cpython", "Python implementation")
    text(python["cache_tag"], "Python cache tag")
    version = text(python["version"], "Python version")
    need(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version) is not None, "Python version spelling")

    probe = exact_dict(
        value["probe"], ("manifest_path", "sha256"), "probe expectation"
    )
    torch = exact_dict(
        value["torch"],
        ("init_manifest_path", "init_sha256", "version"),
        "torch expectation",
    )
    text(torch["version"], "torch version")
    device = exact_dict(
        value["device"], ("index", "physical_uuid", "visibility"), "device expectation"
    )
    visibility = text(device["visibility"], "visibility")
    uuids = canonical_visibility(visibility)
    index = integer(device["index"], "logical index")
    need(0 <= index < len(uuids), "logical index range")
    uuid = canonical_uuid(device["physical_uuid"])
    need(uuids[index] == uuid, "expectation UUID/index")

    declared = (
        (python["manifest_path"], python["sha256"], "python-executable"),
        (probe["manifest_path"], probe["sha256"], "provenance-probe"),
        (torch["init_manifest_path"], torch["init_sha256"], "torch-package-root"),
    )
    role_by_path = {}
    for path_value, sha_value, role in declared:
        path = _member_name(path_value)
        need(path in by_path and path not in role_by_path, "declared runtime path")
        digest = sha256_hex(sha_value, f"{role} hash")
        need(by_path[path]["sha256"] == digest, f"{role} inventory hash")
        role_by_path[path] = role
    pinned_probe = Path(__file__).resolve().parent / "torch_probe_v9.py"
    need(probe["sha256"] == digest_file(pinned_probe), "detached/transplanted torch probe")
    need(PurePosixPath(torch["init_manifest_path"]).name == "__init__.py", "torch init name")
    need(PurePosixPath(torch["init_manifest_path"]).parent.name == "torch", "torch package shape")

    closure = value["runtime_closure"]
    need(type(closure) is list and len(closure) == len(rows), "runtime closure cardinality")
    expected_closure = []
    for row in rows:
        expected_closure.append(
            {
                "manifest_path": row["path"],
                "role": role_by_path.get(row["path"], "python-torch-runtime"),
                "sha256": row["sha256"],
                "size": row["size"],
            }
        )
    need(closure == expected_closure, "runtime closure must exactly cover signed runner tree")
    return value


def validate_torch_report(
    raw_report: bytes,
    expectation: object,
    runner_root: os.PathLike[str] | str,
) -> dict:
    expected = exact_dict(
        expectation,
        (
            "cwd",
            "device",
            "probe",
            "python",
            "runner_inventory_sha256",
            "runner_manifest_sha256",
            "runtime_closure",
            "schema_version",
            "torch",
        ),
        "validated runtime expectation",
    )
    report = canonical_load(raw_report, "torch probe report")
    value = exact_dict(
        report,
        (
            "ignore_environment",
            "index",
            "isolated",
            "loader_identity",
            "module_file",
            "module_identity",
            "module_sha256",
            "no_site",
            "physical_uuid",
            "preimport_absent",
            "python_cache_tag",
            "python_executable",
            "python_executable_sha256",
            "python_implementation",
            "python_version",
            "runner_inventory_sha256",
            "runner_manifest_sha256",
            "schema_version",
            "spec_origin",
            "torch_version",
            "visibility",
        ),
        "torch report schema",
    )
    need(value["schema_version"] == "forkaudit-v9-torch-provenance-v1", "torch report version")
    for key in (
        "ignore_environment",
        "isolated",
        "loader_identity",
        "module_identity",
        "no_site",
        "preimport_absent",
    ):
        need(value[key] is True, key)
    root = Path(runner_root).resolve(strict=True)
    python = expected["python"]
    torch = expected["torch"]
    device = expected["device"]
    expected_path = (root / torch["init_manifest_path"]).resolve(strict=True)
    need(Path(text(value["module_file"], "module file")).resolve(strict=True) == expected_path, "module path")
    need(Path(text(value["spec_origin"], "spec origin")).resolve(strict=True) == expected_path, "spec path")
    need(
        sha256_hex(value["module_sha256"], "module hash")
        == sha256_hex(torch["init_sha256"], "expected module hash")
        == digest_file(expected_path),
        "module digest",
    )
    uuids = canonical_visibility(device["visibility"])
    need(value["visibility"] == device["visibility"], "report visibility")
    logical_index = integer(device["index"], "logical index")
    need(logical_index >= 0 and logical_index < len(uuids), "logical index range")
    need(integer(value["index"], "report logical index") == logical_index, "report index")
    expected_uuid = canonical_uuid(device["physical_uuid"])
    need(expected_uuid == uuids[logical_index], "UUID/visibility binding")
    need(value["physical_uuid"] == expected_uuid, "report physical UUID")
    python_path = (root / python["manifest_path"]).resolve(strict=True)
    need(Path(text(value["python_executable"], "Python executable")).resolve(strict=True) == python_path, "Python executable path")
    need(value["python_executable_sha256"] == python["sha256"] == digest_file(python_path), "Python executable hash")
    need(value["python_implementation"] == python["implementation"], "Python implementation report")
    need(value["python_version"] == python["version"], "Python version report")
    need(value["python_cache_tag"] == python["cache_tag"], "Python cache-tag report")
    need(value["torch_version"] == torch["version"], "torch version report")
    need(value["runner_manifest_sha256"] == expected["runner_manifest_sha256"], "report manifest commitment")
    need(value["runner_inventory_sha256"] == expected["runner_inventory_sha256"], "report inventory commitment")
    return value


def isolated_torch_probe(
    raw_expectation: bytes,
    expected_expectation_sha256: object,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
    expected_runner_manifest_sha256: object,
    expected_runner_inventory_sha256: object,
    timeout_seconds: int = 60,
) -> dict:
    root = Path(runner_root).resolve(strict=True)
    expectation = validate_runtime_expectation(
        raw_expectation,
        expected_expectation_sha256,
        root,
        runner_manifest_path,
        expected_runner_manifest_sha256,
        expected_runner_inventory_sha256,
    )
    python = expectation["python"]
    probe = expectation["probe"]
    torch = expectation["torch"]
    device = expectation["device"]
    python_path = (root / python["manifest_path"]).resolve(strict=True)
    script_path = (root / probe["manifest_path"]).resolve(strict=True)
    init_path = (root / torch["init_manifest_path"]).resolve(strict=True)
    cwd_value, cwd_path = _safe_cwd(root, expectation["cwd"])
    del cwd_value
    argv = [
        str(python_path),
        "-I",
        "-S",
        "-B",
        str(script_path),
        "--torch-init",
        str(init_path),
        "--torch-sha256",
        torch["init_sha256"],
        "--torch-version",
        torch["version"],
        "--visibility",
        device["visibility"],
        "--physical-uuid",
        device["physical_uuid"],
        "--index",
        str(device["index"]),
        "--python-executable",
        str(python_path),
        "--python-sha256",
        python["sha256"],
        "--python-implementation",
        python["implementation"],
        "--python-version",
        python["version"],
        "--python-cache-tag",
        python["cache_tag"],
        "--runner-root",
        str(root),
        "--runner-manifest",
        str(Path(runner_manifest_path).resolve(strict=True)),
        "--runner-manifest-sha256",
        expectation["runner_manifest_sha256"],
        "--runner-inventory-sha256",
        expectation["runner_inventory_sha256"],
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": device["visibility"],
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    need(type(timeout_seconds) is int and 0 < timeout_seconds <= 300, "probe timeout")
    identity_code = (
        "import hashlib,json,os,sys\n"
        "p=os.path.realpath(sys.executable)\n"
        "h=hashlib.sha256(open(p,'rb').read()).hexdigest()\n"
        "r={'cache_tag':sys.implementation.cache_tag,'executable':p,'implementation':sys.implementation.name,"
        "'isolated':sys.flags.isolated==1,'no_site':sys.flags.no_site==1,'sha256':h,"
        "'version':f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'}\n"
        "print(json.dumps(r,sort_keys=True,separators=(',',':')))\n"
    )
    try:
        identity = subprocess.run(
            [str(python_path), "-I", "-S", "-B", "-c", identity_code],
            cwd=cwd_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            check=False,
            timeout=min(timeout_seconds, 10),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Reject("isolated Python identity preflight") from exc
    need(
        identity.returncode == 0 and identity.stderr == b"" and len(identity.stdout) <= 4096,
        "isolated Python identity preflight failure",
    )
    identity_report = canonical_load(identity.stdout, "Python identity report")
    identity_value = exact_dict(
        identity_report,
        (
            "cache_tag",
            "executable",
            "implementation",
            "isolated",
            "no_site",
            "sha256",
            "version",
        ),
        "Python identity report schema",
    )
    need(
        Path(text(identity_value["executable"], "Python identity path")).resolve(strict=True)
        == python_path,
        "Python identity path mismatch",
    )
    need(
        identity_value["implementation"] == python["implementation"]
        and identity_value["version"] == python["version"]
        and identity_value["cache_tag"] == python["cache_tag"]
        and identity_value["sha256"] == python["sha256"]
        and identity_value["isolated"] is True
        and identity_value["no_site"] is True,
        "Python identity mismatch",
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd_path,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Reject("isolated torch probe execution") from exc
    need(
        completed.returncode == 0 and completed.stderr == b"",
        f"isolated torch probe failure rc={completed.returncode} stderr={completed.stderr[:512]!r}",
    )
    need(len(completed.stdout) <= 64 * 1024, "torch report size")
    return validate_torch_report(completed.stdout, expectation, root)


def consumption_record(value: object) -> dict:
    record = exact_dict(
        value,
        ("attempt", "binding_sha256", "run_nonce", "schema_version", "terminal_root"),
        "consumption record",
    )
    attempt = integer(record["attempt"], "consumption attempt")
    need(attempt > 0, "consumption attempt")
    sha256_hex(record["binding_sha256"], "consumption binding")
    sha256_hex(record["run_nonce"], "consumption nonce")
    terminal = text(record["terminal_root"], "consumption terminal root")
    need(Path(terminal).is_absolute() and str(Path(terminal).resolve(strict=False)) == terminal, "consumption terminal path")
    need(record["schema_version"] == "forkaudit-v9-authority-consumption-v1", "consumption version")
    return record


def _string_list(value: object, message: str) -> list[str]:
    need(type(value) is list and bool(value), message)
    out = []
    for item in value:  # type: ignore[union-attr]
        token = text(item, message)
        need("\x00" not in token, message)
        out.append(token)
    return out


def _string_env(value: object) -> dict[str, str]:
    need(type(value) is dict and bool(value), "actual env")
    out = {}
    for key, item in value.items():  # type: ignore[union-attr]
        name = text(key, "actual env key")
        token = text(item, "actual env value")
        need("\x00" not in name + token and name not in out, "actual env item")
        out[name] = token
    return out


def worker_status(value: object, fault_id: str) -> dict:
    row = exact_dict(
        value,
        (
            "actual_argv",
            "actual_argv_schema",
            "actual_cuda_visible_devices",
            "actual_cwd",
            "actual_env",
            "actual_env_schema",
            "actual_executable",
            "actual_physical_uuid",
            "authority_match",
            "death_confirmed",
            "exit_code",
            "fault_id",
            "kill_completed",
            "kill_required",
            "kill_sent",
            "pgid",
            "pid",
            "spawned",
            "spawned_spec_sha256",
            "terminate_sent",
            "wait_completed",
        ),
        "worker status",
    )
    need(row["fault_id"] == fault_id, "worker id")
    for key in (
        "authority_match",
        "death_confirmed",
        "kill_completed",
        "kill_required",
        "kill_sent",
        "spawned",
        "terminate_sent",
        "wait_completed",
    ):
        need(type(row[key]) is bool, f"worker {key}")
    if row["spawned"]:
        pid = integer(row["pid"], "worker pid")
        pgid = integer(row["pgid"], "worker pgid")
        need(pid > 0 and pgid > 0, "worker pid/pgid")
        integer(row["exit_code"], "worker exit code")
        need(row["wait_completed"] is True and row["death_confirmed"] is True, "worker death receipt")
        argv = _string_list(row["actual_argv"], "actual argv")
        need(type(row["actual_argv_schema"]) is list and bool(row["actual_argv_schema"]), "actual argv schema")
        env = _string_env(row["actual_env"])
        need(type(row["actual_env_schema"]) is list and bool(row["actual_env_schema"]), "actual env schema")
        visibility = text(row["actual_cuda_visible_devices"], "actual CUDA visibility")
        canonical_visibility(visibility)
        need(env.get("CUDA_VISIBLE_DEVICES") == visibility, "actual CUDA/env mismatch")
        physical_uuid = canonical_uuid(row["actual_physical_uuid"])
        need(physical_uuid in canonical_visibility(visibility), "actual physical UUID visibility")
        cwd = exact_dict(row["actual_cwd"], ("contract", "dev", "ino", "path"), "actual cwd")
        text(cwd["contract"], "actual cwd contract")
        cwd_path = text(cwd["path"], "actual cwd path")
        need(Path(cwd_path).is_absolute(), "actual cwd absolute")
        need(integer(cwd["dev"], "actual cwd dev") >= 0 and integer(cwd["ino"], "actual cwd ino") > 0, "actual cwd identity")
        executable = exact_dict(
            row["actual_executable"], ("dev", "ino", "path", "sha256"), "actual executable"
        )
        need(text(executable["path"], "actual executable path") == argv[0], "actual executable/argv")
        need(integer(executable["dev"], "actual executable dev") >= 0, "actual executable dev")
        need(integer(executable["ino"], "actual executable ino") > 0, "actual executable ino")
        sha256_hex(executable["sha256"], "actual executable hash")
        spec = {
            "argv": argv,
            "argv_schema": row["actual_argv_schema"],
            "cwd": cwd_path,
            "cwd_contract": cwd["contract"],
            "env": env,
            "env_schema": row["actual_env_schema"],
            "fault_id": fault_id,
        }
        need(
            sha256_hex(row["spawned_spec_sha256"], "spawned spec hash")
            == digest_bytes(canonical_bytes(spec)),
            "actual spawned spec commitment",
        )
    else:
        need(
            row["actual_argv"] is None
            and row["actual_argv_schema"] is None
            and row["actual_cuda_visible_devices"] is None
            and row["actual_cwd"] is None
            and row["actual_env"] is None
            and row["actual_env_schema"] is None
            and row["actual_executable"] is None
            and row["actual_physical_uuid"] is None
            and row["authority_match"] is False
            and row["exit_code"] is None
            and row["pgid"] is None
            and row["pid"] is None
            and row["spawned_spec_sha256"] is None
            and row["wait_completed"] is False
            and row["death_confirmed"] is False,
            "unspawned worker receipt",
        )
    need(row["terminate_sent"] == row["kill_required"], "worker terminate aggregation")
    need(not row["kill_sent"] or row["kill_required"], "worker kill escalation")
    expected_completed = (not row["kill_required"]) or (
        row["wait_completed"] and row["death_confirmed"]
    )
    need(row["kill_completed"] == expected_completed, "worker kill completion")
    return row


def validate_spawn_provenance(value: object) -> dict:
    provenance = exact_dict(value, SPAWN_PROVENANCE_KEYS, "spawn provenance")
    for key in SPAWN_PROVENANCE_KEYS:
        sha256_hex(provenance[key], key)
    return provenance


def lifecycle_gate(value: object, require_success: bool = False) -> dict:
    need(type(require_success) is bool, "success policy")
    gate = exact_dict(
        value,
        (
            "attempt",
            "consumption_sha256",
            "inventory_verified",
            "kill_completion",
            "post_rehash_verified",
            "provenance",
            "provenance_verified",
            "receipts_verified",
            "run_nonce",
            "schema_version",
            "terminal_root",
            "verification_complete",
            "workers",
        ),
        "lifecycle gate",
    )
    need(gate["schema_version"] == "forkaudit-v9-lifecycle-gate-v1", "gate version")
    need(integer(gate["attempt"], "gate attempt") > 0, "gate attempt")
    sha256_hex(gate["run_nonce"], "gate nonce")
    sha256_hex(gate["consumption_sha256"], "gate consumption")
    terminal_root = text(gate["terminal_root"], "gate terminal root")
    need(Path(terminal_root).is_absolute(), "gate terminal root absolute")
    for key in (
        "inventory_verified",
        "post_rehash_verified",
        "provenance_verified",
        "receipts_verified",
        "verification_complete",
    ):
        need(type(gate[key]) is bool, key)
    provenance = validate_spawn_provenance(gate["provenance"])
    need(provenance["consumption_sha256"] == gate["consumption_sha256"], "gate consumption/provenance")
    kill = exact_dict(gate["kill_completion"], ("completed", "errors", "required"), "kill completion")
    need(type(kill["required"]) is bool and type(kill["completed"]) is bool, "kill completion status")
    need(type(kill["errors"]) is list, "kill errors")
    for error in kill["errors"]:
        text(error, "kill error")
    workers = gate["workers"]
    need(type(workers) is list and len(workers) == len(TERM_IDS), "worker cardinality")
    parsed = [worker_status(row, fault_id) for fault_id, row in zip(TERM_IDS, workers)]
    need(kill["required"] == any(row["kill_required"] for row in parsed), "overall kill required")
    need(
        kill["completed"] == (all(row["kill_completed"] for row in parsed) and kill["errors"] == []),
        "overall kill completed",
    )
    receipts_ok = all(
        (not row["spawned"])
        or (row["wait_completed"] and row["death_confirmed"] and type(row["exit_code"]) is int)
        for row in parsed
    )
    need(gate["receipts_verified"] == receipts_ok, "receipts aggregation")
    verified = (
        gate["inventory_verified"]
        and gate["post_rehash_verified"]
        and gate["provenance_verified"]
        and gate["receipts_verified"]
        and kill["completed"]
    )
    need(gate["verification_complete"] == verified, "verification aggregation")
    if require_success:
        need(
            verified
            and all(row["spawned"] for row in parsed)
            and all(row["authority_match"] for row in parsed)
            and all(row["pgid"] == row["pid"] for row in parsed)
            and all(not row["kill_required"] for row in parsed)
            and all(row["exit_code"] == 0 for row in parsed),
            "lifecycle not successful",
        )
    return gate


def terminal_record(value: object, fault_id: str) -> dict:
    terminal = exact_dict(
        value,
        (
            "fault_id",
            "lifecycle_receipt",
            "lifecycle_receipt_sha256",
            "post_hashes",
            "pre_hashes",
            "provenance",
            "provenance_sha256",
            "reason",
            "schema_version",
            "signal",
            "status",
        ),
        "terminal schema",
    )
    need(terminal["schema_version"] == "forkaudit-v9-terminal-v1", "terminal version")
    need(terminal["fault_id"] == fault_id, "terminal id")
    status_value = terminal["status"]
    need(status_value in ("success", "failure"), "terminal status")
    text(terminal["reason"], "terminal reason")
    signal_value = terminal["signal"]
    need(signal_value is None or (type(signal_value) is int and signal_value in (2, 15)), "terminal signal")
    receipt = lifecycle_gate(terminal["lifecycle_receipt"], require_success=status_value == "success")
    need(
        sha256_hex(terminal["lifecycle_receipt_sha256"], "lifecycle receipt hash")
        == digest_bytes(canonical_bytes(receipt)),
        "lifecycle receipt commitment",
    )
    provenance = validate_spawn_provenance(terminal["provenance"])
    need(provenance == receipt["provenance"], "terminal/gate provenance mismatch")
    need(
        sha256_hex(terminal["provenance_sha256"], "provenance hash")
        == digest_bytes(canonical_bytes(provenance)),
        "terminal provenance commitment",
    )
    pre = validate_hash_map(terminal["pre_hashes"], allow_null=status_value == "failure")
    post = validate_hash_map(terminal["post_hashes"], allow_null=status_value == "failure")
    if status_value == "success":
        need(terminal["reason"] == "success" and signal_value is None and pre == post, "terminal success gate")
    return terminal


def validate_terminal_tree(
    root: os.PathLike[str] | str,
    expected_pre_hashes: object,
    expected_post_hashes: object,
    expected_status: str,
) -> None:
    need(expected_status in ("success", "failure"), "expected status")
    pre = validate_hash_map(expected_pre_hashes, allow_null=expected_status == "failure")
    post = validate_hash_map(expected_post_hashes, allow_null=expected_status == "failure")
    directory = Path(root)
    directory_stat = os.lstat(directory)
    need(stat.S_ISDIR(directory_stat.st_mode) and not stat.S_ISLNK(directory_stat.st_mode), "terminal root")
    expected_names = {"AUTHORIZED_CONSUMPTION.json"} | {
        f"{fault_id}.terminal.json" for fault_id in TERM_IDS
    }
    entries = list(os.scandir(directory))
    need({entry.name for entry in entries} == expected_names, "exact terminal names")
    inodes = set()
    for entry in entries:
        item_stat = entry.stat(follow_symlinks=False)
        need(stat.S_ISREG(item_stat.st_mode) and item_stat.st_nlink == 1, "terminal regular unique")
        key = (item_stat.st_dev, item_stat.st_ino)
        need(key not in inodes, "terminal duplicate inode")
        inodes.add(key)
    consumption_raw, _ = _read_regular(directory / "AUTHORIZED_CONSUMPTION.json")
    consumption = consumption_record(canonical_load(consumption_raw, "consumption"))
    consumption_sha = digest_bytes(consumption_raw)
    need(consumption["terminal_root"] == str(directory.resolve(strict=True)), "consumption root")
    common_receipt_sha = None
    common_provenance_sha = None
    for fault_id in TERM_IDS:
        raw, _ = _read_regular(directory / f"{fault_id}.terminal.json")
        parsed = terminal_record(canonical_load(raw, "terminal"), fault_id)
        need(parsed["status"] == expected_status, "terminal expected status")
        need(parsed["pre_hashes"] == pre and parsed["post_hashes"] == post, "terminal hash maps")
        receipt = parsed["lifecycle_receipt"]
        need(
            receipt["attempt"] == consumption["attempt"]
            and receipt["run_nonce"] == consumption["run_nonce"]
            and receipt["terminal_root"] == consumption["terminal_root"]
            and receipt["consumption_sha256"] == consumption_sha,
            "terminal/consumption authority",
        )
        if common_receipt_sha is None:
            common_receipt_sha = parsed["lifecycle_receipt_sha256"]
            common_provenance_sha = parsed["provenance_sha256"]
        need(parsed["lifecycle_receipt_sha256"] == common_receipt_sha, "terminal receipt disagreement")
        need(parsed["provenance_sha256"] == common_provenance_sha, "terminal provenance disagreement")
