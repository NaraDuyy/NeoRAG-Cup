"""Trích từ điển viết tắt / teencode từ hai dataset chuẩn hóa văn bản tiếng Việt công khai.

    python scripts/build_abbreviations.py --yammdd <thư mục CSV> --vilexnorm <thư mục CSV>

Nguồn (không commit dữ liệu thô, chỉ commit từ điển đã trích):
- yammdd/vietnamese-error-correction-corpus (Hugging Face, MIT): cột input/target, nhãn do LLM sinh.
- ViLexNorm (Nguyen et al., EACL 2024, CC BY-NC-SA 4.0): cột original/normalized, người gắn nhãn.

Mỗi cặp câu được căn theo âm tiết bỏ dấu; một token gốc bị thay bằng cụm dài hơn và khác nó sau khi
bỏ dấu được coi là một viết tắt. Ánh xạ có trong ViLexNorm, hoặc gặp ít nhất --min-count lần trong
yammdd, mới được giữ. Kết quả ghi ra src/neorag/resources/abbreviations.tsv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from neorag.retrieval import strip_accents
from neorag.text import tokenize

OUTPUT = Path(__file__).resolve().parents[1] / "src" / "neorag" / "resources" / "abbreviations.tsv"

# Mục soạn tay: viết tắt riêng của các CLB và cách gọi thường gặp trong câu hỏi
MANUAL = {
    "mm": ["multimedia"], "mpc": ["multimedia"], "dpt": ["đa phương tiện multimedia"],
    "lt": ["lập trình"], "pro": ["proptit"], "tv": ["thành viên"],
    "ttv": ["tuyển thành viên"], "nk": ["nhiệm kỳ"], "cn": ["chủ nhiệm", "chủ nhật"],
    "bcn": ["ban chủ nhiệm"], "hv": ["học viện", "học viên"], "dk": ["đăng ký", "điều kiện"],
    "dc": ["được"], "sv": ["sinh viên"], "j": ["gì"], "v": ["vậy"], "ko": ["không"],
    "k": ["không"], "ntn": ["như thế nào"], "sk": ["sự kiện"], "vs": ["với"],
    "tt": ["thông tin", "truyền thông", "thực tập"], "hd": ["hoạt động", "hướng dẫn"],
}  # fmt: skip
# Từ tục tĩu: bỏ mọi viết tắt có nghĩa chứa chúng (so khớp có dấu, vì bỏ dấu sẽ trùng từ thường)
_VULGAR = "địt lồn cặc buồi đéo đụ đếch"
_VULGAR_TOKENS = "dm dmm dcm dkm vl vcl vkl cc clgt cl cmm vcc vc dml"
VULGAR, VULGAR_TOKENS = frozenset(_VULGAR.split()), frozenset(_VULGAR_TOKENS.split())
VULGAR_PHRASES = ("súc vật", "chó chết", "mặt lồn")
# Chữ thay thế hay gặp trong teencode: chữ của viết tắt -> các chữ nó có thể thay trong từ chuẩn
_TEENCODE = {"k": "kc", "f": "fp", "j": "jgd", "z": "zdvg", "w": "wqu", "x": "xaus"}


def read_pairs(folder: Path, src: str, dst: str) -> list[tuple[str, str]]:
    pairs = []
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as f:
            pairs += [(r[src], r[dst]) for r in csv.DictReader(f) if r.get(src) and r.get(dst)]
    return pairs


def extract(pairs: list[tuple[str, str]]) -> Counter[tuple[str, str]]:
    """Đếm các ánh xạ token (bỏ dấu) -> cụm chuẩn (có dấu) qua phép căn chuỗi âm tiết."""
    found: Counter[tuple[str, str]] = Counter()
    for source, target in pairs:
        src = [strip_accents(t) for t in tokenize(source)]
        tgt = tokenize(target)
        tgt_plain = [strip_accents(t) for t in tgt]
        matcher = SequenceMatcher(None, src, tgt_plain, autojunk=False)
        if matcher.ratio() < 0.5:
            continue
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op != "replace" or i2 - i1 != 1 or not 1 <= j2 - j1 <= 4:
                continue
            token, phrase = src[i1], " ".join(tgt[j1:j2])
            plain = strip_accents(phrase)
            if plain == token or not token.isalnum():
                continue
            # Viết tắt phải ngắn hơn cụm chuẩn; ngược lại thường là lỗi gõ
            if j2 - j1 == 1 and len(token) >= len(plain):
                continue
            found[token, phrase] += 1
    return found


def is_typo(token: str, plain: str) -> bool:
    """Token chỉ thiếu 1 ký tự so với một âm tiết (lic -> lich) thường là lỗi gõ, không phải viết tắt."""
    return " " not in plain and len(plain) - len(token) == 1 and _is_subsequence(token, plain)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def matches_first_letter(token: str, plain: str) -> bool:
    """Nghĩa nhiều âm tiết phải bắt đầu bằng chữ đầu của viết tắt (tính cả chữ thay thế teencode).

    Loại các nhãn lỗi do phép căn chuỗi gộp nhầm từ đứng trước: `dc` -> "bang duoc",
    `post` -> "bai dang". Nghĩa một âm tiết thì không xét, vì teencode biến đổi chữ tự do
    (`iu` -> "yêu", `mk` -> "mình", `bh` -> "bây giờ" vẫn hợp lệ).
    """
    return " " not in plain or plain[0] in _TEENCODE.get(token[0], token[0])


def syllable_counts(pairs: list[tuple[str, str]]) -> Counter[str]:
    """Tần suất âm tiết (bỏ dấu) trong các câu chuẩn: dùng làm từ vựng tiếng Việt."""
    return Counter(strip_accents(t) for _, target in pairs for t in tokenize(target))


def build(
    yammdd: Counter, vilexnorm: Counter, vocab: Counter, min_count: int
) -> dict[str, list[tuple[str, str]]]:
    """viết tắt -> [(nghĩa có dấu, nguồn)], nghĩa phổ biến trước."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)  # token -> nghĩa bỏ dấu -> số lần
    spelling: dict[str, Counter[str]] = defaultdict(Counter)  # nghĩa bỏ dấu -> cách viết có dấu
    origin: dict[tuple[str, str], str] = {}
    rejected: list[tuple[str, str]] = []
    for (token, phrase), n in (yammdd + vilexnorm).items():
        in_vln = vilexnorm[token, phrase] > 0
        if not in_vln and yammdd[token, phrase] < min_count:
            continue
        plain = strip_accents(phrase)
        if (
            token in VULGAR_TOKENS
            or VULGAR & set(phrase.split())
            or any(p in phrase for p in VULGAR_PHRASES)
        ):
            continue
        # Token có chữ số, hoặc bản thân là một âm tiết phổ biến (nhin, ca), không phải viết tắt
        if not token.isalpha() or vocab[token] >= 50:
            continue
        # Nghĩa chứa âm tiết hiếm (khongo, cunh) là nhãn lỗi
        if any(vocab[s] < 20 for s in plain.split()):
            continue
        if not in_vln and is_typo(token, plain):
            continue
        if not matches_first_letter(token, plain):
            rejected.append((token, phrase))
            continue
        counts[token][plain] += n
        spelling[plain][phrase] += n
        old = origin.get((token, plain))
        new = "both" if in_vln and yammdd[token, phrase] else ("vilexnorm" if in_vln else "yammdd")
        origin[token, plain] = new if old in (None, new) else "both"
    table = {
        token: [(spelling[p].most_common(1)[0][0], origin[token, p]) for p, _ in c.most_common()]
        for token, c in counts.items()
    }
    for token, phrases in MANUAL.items():
        manual = {strip_accents(p) for p in phrases}
        rest = [s for s in table.get(token, []) if strip_accents(s[0]) not in manual]
        table[token] = [(p, "manual") for p in phrases] + rest
    for n in range(1, 13):
        phrases = [f"tháng {n}", f"thứ {n}"] if 2 <= n <= 7 else [f"tháng {n}"]
        table[f"t{n}"] = [(p, "manual") for p in phrases]
    print(
        f"Loại {len(rejected)} nghĩa nhiều âm tiết không bắt đầu bằng chữ đầu của viết tắt:",
        rejected,
    )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yammdd", type=Path, required=True)
    parser.add_argument("--vilexnorm", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    yammdd_pairs = read_pairs(args.yammdd, "input", "target")
    vilexnorm_pairs = read_pairs(args.vilexnorm, "original", "normalized")
    vocab = syllable_counts(yammdd_pairs + vilexnorm_pairs)
    table = build(extract(yammdd_pairs), extract(vilexnorm_pairs), vocab, args.min_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        f.write(
            "# Từ điển viết tắt (cột 1 đã bỏ dấu), sinh bởi scripts/build_abbreviations.py.\n"
            "# Nguồn: yammdd/vietnamese-error-correction-corpus (Hugging Face, MIT);\n"
            "# ViLexNorm (Nguyen et al., EACL 2024, https://github.com/ngxtnhi/ViLexNorm,\n"
            "# CC BY-NC-SA 4.0); manual = soạn tay. Cột: viết tắt, các nghĩa (|), nguồn từng nghĩa.\n"
        )
        for token in sorted(table):
            phrases, origins = zip(*table[token], strict=True)
            f.write(f"{token}\t{'|'.join(phrases)}\t{'|'.join(origins)}\n")
    print(f"{len(table)} viết tắt -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
