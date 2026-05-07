import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from yesno_utils import (  # noqa: E402
    derive_follow_label,
    invert_yes_no,
    make_conflict_prompt,
    make_support_prompt,
    make_nc_prompt,
    read_jsonl,
    wrap_as_chat,
    write_json,
    write_jsonl,
)

YES = "yes"
NO = "no"
FOLLOW_MAP = {
    "resist": 0,
    "follow_conflict": 1,
    "unknown": 2,
}


def opposite_label(label):
    return invert_yes_no(label)


def load_outcome_map(conflict_positions_path, position):
    if not conflict_positions_path:
        return {}
    rows = read_jsonl(conflict_positions_path)
    outcome_map = {}
    for row in rows:
        if row.get("position") != position:
            continue
        outcome_map[row.get("pair_id")] = row
    return outcome_map
