from __future__ import annotations

import binascii
import hashlib
import importlib.util
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
    "963171ca22062440c9a4c80feba30998f597105d9b37489c1a1828b208cb1e6b"
)
BINDING_DOMAIN = b"forkaudit-v6-operator-binding-payload-v1\x00"
TERM_IDS = tuple(f"V6F{x:02d}" for x in range(1, 9))
HASH_KEYS = (
    "archive_sha256",
    "source_ledger_sha256",
    "snapshot_sha256",
    "snapshot_inventory_sha256",
    "runner_inventory_sha256",
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
    rows, inventory_hash = _inventory(files, "forkaudit-v6-snapshot-inventory-v1")
    commitment = hashlib.sha256(b"forkaudit-v6-snapshot-bytes-v1\x00")
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


def validate_runner_tree(root: os.PathLike[str] | str, manifest: object) -> tuple[str, list[dict]]:
    expected = _manifest_rows(manifest)
    files = _scan_closed_tree(root)
    actual, inventory_hash = _inventory(files, "forkaudit-v6-runner-inventory-v1")
    need(actual == expected, "runner inventory mismatch")
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
    runner_manifest: object,
) -> dict:
    snapshot_hash, snapshot_inventory_hash, _ = snapshot_commitments(snapshot_root)
    runner_inventory_hash, _ = validate_runner_tree(runner_root, runner_manifest)
    return {
        "archive_sha256": digest_file(archive_path),
        "source_ledger_sha256": digest_file(source_ledger_path),
        "snapshot_sha256": snapshot_hash,
        "snapshot_inventory_sha256": snapshot_inventory_hash,
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
    need(ledger_dict["schema_version"] == "forkaudit-v6-source-ledger-v1", "ledger version")
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
    need(root_dict["schema_version"] == "forkaudit-v6-operator-trust-root-v1", "trust root version")
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
    torch_expectation_path: os.PathLike[str] | str,
) -> dict:
    _compiled_trust_root()
    binding = canonical_load(raw_binding, "signed operator binding")
    outer = exact_dict(binding, ("payload", "schema_version", "signature"), "binding schema")
    need(outer["schema_version"] == "forkaudit-v6-signed-operator-binding-v1", "binding version")
    payload = exact_dict(
        outer["payload"],
        (
            "approved_archive_sha256",
            "approved_execution_contract_sha256",
            "approved_snapshot_inventory_sha256",
            "approved_snapshot_sha256",
            "approved_source_ledger_sha256",
            "approved_torch_expectation_sha256",
            "operator_id",
            "published_uri",
            "schema_version",
            "trust_root_id",
        ),
        "binding payload",
    )
    signature = exact_dict(outer["signature"], ("algorithm", "key_id", "signature_hex"), "binding signature")
    need(payload["schema_version"] == "forkaudit-v6-operator-binding-payload-v1", "payload version")
    need(payload["trust_root_id"] == TRUST_ROOT_ID, "payload trust root")
    text(payload["operator_id"], "operator id")
    _published_uri(payload["published_uri"])
    for key in (
        "approved_archive_sha256",
        "approved_execution_contract_sha256",
        "approved_snapshot_inventory_sha256",
        "approved_snapshot_sha256",
        "approved_source_ledger_sha256",
        "approved_torch_expectation_sha256",
    ):
        sha256_hex(payload[key], key)
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
    need(
        digest_file(execution_contract_path) == payload["approved_execution_contract_sha256"],
        "execution contract digest",
    )
    need(
        digest_file(torch_expectation_path) == payload["approved_torch_expectation_sha256"],
        "torch expectation digest",
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


_PATH_WORDS = (
    "config",
    "file",
    "path",
    "dir",
    "root",
    "model",
    "checkpoint",
    "manifest",
    "output",
    "input",
    "script",
    "tokenizer",
    "cache",
)
_PATH_SUFFIXES = (
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".sh",
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".txt",
    ".csv",
)


def _path_option(option: str) -> bool:
    normalized = option.lstrip("-").lower().replace("_", "-")
    return any(word in normalized.split("-") for word in _PATH_WORDS)


def _looks_like_path(token: str) -> bool:
    lower = token.lower()
    return (
        "/" in token
        or "\\" in token
        or token.startswith((".", "~"))
        or lower.endswith(_PATH_SUFFIXES)
    )


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


def validate_execution_contract(
    raw_contract: bytes,
    expected_contract_sha256: object,
    runner_root: os.PathLike[str] | str,
    runner_manifest: object,
) -> dict:
    need(digest_bytes(raw_contract) == sha256_hex(expected_contract_sha256, "contract anchor"), "contract digest")
    contract = canonical_load(raw_contract, "execution contract")
    value = exact_dict(
        contract,
        ("argv", "cwd", "env", "path_bindings", "schema_version"),
        "execution contract schema",
    )
    need(value["schema_version"] == "forkaudit-v6-execution-contract-v1", "contract version")
    argv = value["argv"]
    need(type(argv) is list and bool(argv), "argv")
    for arg in argv:
        text(arg, "argv item")
        need("\x00" not in arg, "argv NUL")
    env = value["env"]
    need(type(env) is dict, "env")
    for key, item in env.items():
        need(type(key) is str and bool(key) and type(item) is str and "\x00" not in key + item, "env item")
    forbidden = {"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"}
    need(not (set(env) & forbidden), "ambient import/loader env")
    need(env.get("PYTHONHASHSEED") == "0" and env.get("PYTHONNOUSERSITE") == "1", "deterministic Python env")
    canonical_visibility(env.get("CUDA_VISIBLE_DEVICES"))

    root = Path(runner_root)
    root_stat = os.lstat(root)
    need(stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode), "runner root")
    _, cwd_path = _safe_cwd(root, value["cwd"])
    _, actual_rows = validate_runner_tree(root, runner_manifest)
    by_path = {row["path"]: row for row in actual_rows}

    path_bindings = value["path_bindings"]
    need(type(path_bindings) is list and bool(path_bindings), "path bindings")
    bound: dict[int, tuple[str, object]] = {}
    last_index = -1
    for item in path_bindings:
        row = exact_dict(item, ("argv_index", "manifest_path", "option"), "path binding row")
        index = integer(row["argv_index"], "argv index")
        need(0 <= index < len(argv) and index > last_index, "path binding order/duplicate")
        last_index = index
        manifest_path = _member_name(row["manifest_path"])
        need(manifest_path in by_path, "path not in runner inventory")
        option = row["option"]
        need(option is None or (type(option) is str and option.startswith("-") and "=" not in option), "path option")
        token = argv[index]
        if option is None:
            path_value = token
        elif token.startswith(option + "="):
            path_value = token[len(option) + 1 :]
            need(bool(path_value), "empty option path")
        else:
            need(index > 0 and argv[index - 1] == option, "separate option path")
            path_value = token
        need(not path_value.startswith("~") and "\\" not in path_value, "path spelling")
        candidate = Path(path_value)
        resolved = candidate if candidate.is_absolute() else cwd_path / candidate
        try:
            relative = resolved.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
        except (OSError, ValueError) as exc:
            raise Reject("path escapes runner root") from exc
        need(relative == manifest_path, "path binding target")
        target_stat = os.lstat(resolved)
        need(stat.S_ISREG(target_stat.st_mode) and target_stat.st_nlink == 1, "bound path regular unique")
        need(digest_file(resolved) == by_path[manifest_path]["sha256"], "bound path content")
        bound[index] = (manifest_path, option)

    detected = {0}
    for index, token in enumerate(argv):
        if index == 0 or _looks_like_path(token):
            detected.add(index)
        if token.startswith("-") and "=" in token:
            option, path_value = token.split("=", 1)
            if _path_option(option) or _looks_like_path(path_value):
                detected.add(index)
        if index > 0 and argv[index - 1].startswith("-") and _path_option(argv[index - 1]):
            detected.add(index)
    need(set(bound) == detected, "nonexhaustive path bindings")
    return value


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


def validate_torch_report(
    raw_report: bytes,
    expected_init_path: os.PathLike[str] | str,
    expected_init_sha256: object,
    visibility: object,
    physical_uuid: object,
    index: object,
) -> dict:
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
            "schema_version",
            "spec_origin",
            "visibility",
        ),
        "torch report schema",
    )
    need(value["schema_version"] == "forkaudit-v6-torch-provenance-v1", "torch report version")
    for key in (
        "ignore_environment",
        "isolated",
        "loader_identity",
        "module_identity",
        "no_site",
        "preimport_absent",
    ):
        need(value[key] is True, key)
    expected_path = Path(expected_init_path).resolve(strict=True)
    need(Path(text(value["module_file"], "module file")).resolve(strict=True) == expected_path, "module path")
    need(Path(text(value["spec_origin"], "spec origin")).resolve(strict=True) == expected_path, "spec path")
    need(
        sha256_hex(value["module_sha256"], "module hash")
        == sha256_hex(expected_init_sha256, "expected module hash")
        == digest_file(expected_path),
        "module digest",
    )
    uuids = canonical_visibility(visibility)
    need(value["visibility"] == visibility, "report visibility")
    logical_index = integer(index, "logical index")
    need(logical_index >= 0 and logical_index < len(uuids), "logical index range")
    need(value["index"] == logical_index, "report index")
    expected_uuid = canonical_uuid(physical_uuid)
    need(expected_uuid == uuids[logical_index], "UUID/visibility binding")
    need(value["physical_uuid"] == expected_uuid, "report physical UUID")
    return value


def isolated_torch_probe(
    python_executable: os.PathLike[str] | str,
    probe_script: os.PathLike[str] | str,
    torch_init_path: os.PathLike[str] | str,
    expected_init_sha256: object,
    visibility: object,
    physical_uuid: object,
    index: object,
    cwd: os.PathLike[str] | str,
    runner_root: os.PathLike[str] | str,
    runner_manifest: object,
    timeout_seconds: int = 60,
) -> dict:
    root = Path(runner_root).resolve(strict=True)
    _, rows = validate_runner_tree(root, runner_manifest)
    allowed = {row["path"]: row for row in rows}

    def bound(path: os.PathLike[str] | str) -> Path:
        resolved = Path(path).resolve(strict=True)
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise Reject("probe path outside inventory") from exc
        need(relative in allowed and digest_file(resolved) == allowed[relative]["sha256"], "probe path binding")
        return resolved

    python_path = bound(python_executable)
    script_path = bound(probe_script)
    init_path = bound(torch_init_path)
    pinned_probe = Path(__file__).resolve().parent / "torch_probe_v6.py"
    need(
        digest_file(script_path) == digest_file(pinned_probe),
        "detached/transplanted torch probe",
    )
    cwd_path = Path(cwd).resolve(strict=True)
    try:
        cwd_path.relative_to(root)
    except ValueError as exc:
        raise Reject("probe cwd outside runner") from exc
    uuid = canonical_uuid(physical_uuid)
    visible = text(visibility, "visibility")
    logical_index = integer(index, "index")
    uuids = canonical_visibility(visible)
    need(0 <= logical_index < len(uuids) and uuids[logical_index] == uuid, "probe UUID/index")
    expected_sha = sha256_hex(expected_init_sha256, "torch init hash")
    need(digest_file(init_path) == expected_sha, "torch init prehash")
    argv = [
        str(python_path),
        "-I",
        "-S",
        "-B",
        str(script_path),
        "--torch-init",
        str(init_path),
        "--torch-sha256",
        expected_sha,
        "--visibility",
        visible,
        "--physical-uuid",
        uuid,
        "--index",
        str(logical_index),
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": visible,
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    need(type(timeout_seconds) is int and 0 < timeout_seconds <= 300, "probe timeout")
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
    return validate_torch_report(
        completed.stdout, init_path, expected_sha, visible, uuid, logical_index
    )


def worker_status(value: object, fault_id: str) -> dict:
    status_value = exact_dict(
        value,
        (
            "death_confirmed",
            "exit_code",
            "fault_id",
            "kill_completed",
            "kill_required",
            "spawned",
        ),
        "worker status",
    )
    need(status_value["fault_id"] == fault_id, "worker id")
    need(status_value["spawned"] is True, "worker spawned")
    for key in ("death_confirmed", "kill_completed", "kill_required"):
        need(type(status_value[key]) is bool, f"worker {key}")
    exit_code = integer(status_value["exit_code"], "worker exit code")
    need(exit_code >= 0, "worker exit code")
    if status_value["kill_required"]:
        need(status_value["kill_completed"] is True, "worker kill incomplete")
    return status_value


def lifecycle_gate(value: object) -> dict:
    gate = exact_dict(
        value,
        (
            "inventory_verified",
            "kill_completion",
            "post_rehash_verified",
            "receipts_verified",
            "schema_version",
            "verification_complete",
            "workers",
        ),
        "lifecycle gate",
    )
    need(gate["schema_version"] == "forkaudit-v6-lifecycle-gate-v1", "gate version")
    for key in (
        "inventory_verified",
        "post_rehash_verified",
        "receipts_verified",
        "verification_complete",
    ):
        need(gate[key] is True, key)
    kill = exact_dict(gate["kill_completion"], ("completed", "errors", "required"), "kill completion")
    need(type(kill["required"]) is bool and kill["completed"] is True, "kill completion status")
    need(type(kill["errors"]) is list and kill["errors"] == [], "kill errors")
    workers = gate["workers"]
    need(type(workers) is list and len(workers) == len(TERM_IDS), "worker cardinality")
    for fault_id, status_value in zip(TERM_IDS, workers):
        status_row = worker_status(status_value, fault_id)
        need(
            status_row["death_confirmed"] is True
            and status_row["exit_code"] == 0
            and status_row["kill_completed"] is True,
            "worker not successful",
        )
    return gate


def terminal_record(value: object, fault_id: str) -> dict:
    terminal = exact_dict(
        value,
        (
            "fault_id",
            "post_hashes",
            "pre_hashes",
            "reason",
            "schema_version",
            "signal",
            "status",
        ),
        "terminal schema",
    )
    need(terminal["schema_version"] == "forkaudit-v6-terminal-v1", "terminal version")
    need(terminal["fault_id"] == fault_id, "terminal id")
    status_value = terminal["status"]
    need(status_value in ("success", "failure"), "terminal status")
    text(terminal["reason"], "terminal reason")
    signal_value = terminal["signal"]
    need(signal_value is None or signal_value in (2, 15), "terminal signal")
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
    expected_names = {f"{fault_id}.terminal.json" for fault_id in TERM_IDS}
    entries = list(os.scandir(directory))
    need({entry.name for entry in entries} == expected_names, "exact terminal names")
    inodes = set()
    for entry in entries:
        item_stat = entry.stat(follow_symlinks=False)
        need(stat.S_ISREG(item_stat.st_mode) and item_stat.st_nlink == 1, "terminal regular unique")
        key = (item_stat.st_dev, item_stat.st_ino)
        need(key not in inodes, "terminal duplicate inode")
        inodes.add(key)
    for fault_id in TERM_IDS:
        raw, _ = _read_regular(directory / f"{fault_id}.terminal.json")
        terminal = canonical_load(raw, "terminal")
        parsed = terminal_record(terminal, fault_id)
        need(parsed["status"] == expected_status, "terminal expected status")
        need(parsed["pre_hashes"] == pre and parsed["post_hashes"] == post, "terminal hash maps")
