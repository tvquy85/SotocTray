import argparse
import pathlib
import sys

FORBIDDEN = [
    "first framework",
    "đầu tiên",
    "sota",
    "state-of-the-art",
    "thắng tuyệt đối",
    "perfect shield",
    "tấm khiên hoàn hảo",
    "siêu lợi nhuận",
    "happy path",
    "guaranteed",
    "hoàn hảo",
]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="paper")
    args = parser.parse_args()

    bad = []
    for path in pathlib.Path(args.root).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in FORBIDDEN:
            if phrase.lower() in text:
                bad.append((str(path), phrase))

    if bad:
        for path, phrase in bad:
            print(f"FORBIDDEN: {path}: {phrase}")
        sys.exit(1)
    print("Claim audit passed")


if __name__ == "__main__":
    main()
