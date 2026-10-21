import re
import unicodedata
from collections.abc import Iterable, Mapping


YES_VARIANTS = {"yes", "y", "yeah", "yep", "true", "correct", "affirmative"}
NO_VARIANTS = {"no", "n", "nope", "false", "incorrect", "negative"}
FAILED_OUTPUTS = {"error", "failed", "failure", "none", "null", "n/a"}

COCO_CATEGORIES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

_COCO_ALIASES = {category: category for category in COCO_CATEGORIES}
_COCO_ALIASES.update(
    {
        "people": "person", "persons": "person", "man": "person", "men": "person",
        "woman": "person", "women": "person", "boy": "person", "boys": "person",
        "girl": "person", "girls": "person", "bike": "bicycle", "bikes": "bicycle",
        "motorbike": "motorcycle", "motorbikes": "motorcycle", "plane": "airplane",
        "planes": "airplane", "aeroplane": "airplane", "sofa": "couch", "settee": "couch",
        "television": "tv", "televisions": "tv", "cellphone": "cell phone",
        "cellphones": "cell phone", "mobile phone": "cell phone", "mobile phones": "cell phone",
        "hair dryer": "hair drier", "doughnut": "donut", "doughnuts": "donut",
    }
)


def _plural_forms(phrase):
    if phrase.endswith("s"):
        return ()
    if phrase.endswith(("ch", "sh", "x", "z")):
        return (phrase + "es",)
    if phrase.endswith("y") and len(phrase) > 1 and phrase[-2] not in "aeiou":
        return (phrase[:-1] + "ies",)
    return (phrase + "s",)


for _category in COCO_CATEGORIES:
    for _plural in _plural_forms(_category):
        _COCO_ALIASES.setdefault(_plural, _category)

_COCO_ALIAS_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(value) for value in sorted(_COCO_ALIASES, key=len, reverse=True)) + r")\b"
)


def normalize_text(value, normalization="strict"):
    if normalization not in {"strict", "lenient"}:
        raise ValueError(f"invalid answer normalization: {normalization}")
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    text = " ".join(text.split())
    if normalization == "lenient":
        text = re.sub(r"^(the|a|an)\s+", "", text)
    return text


def extract_answer(value):
    text = str(value or "").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def normalize_yes_no(value):
    for token in normalize_text(extract_answer(value), "lenient").split():
        if token in YES_VARIANTS:
            return "yes"
        if token in NO_VARIANTS:
            return "no"
    return None


def extract_coco_objects(value):
    """Return canonical COCO object categories mentioned by ``value``."""
    text = normalize_text(extract_answer(value), "lenient")
    return {_COCO_ALIASES[match.group(0)] for match in _COCO_ALIAS_PATTERN.finditer(text)}


def _caption_ground_truth_objects(ground_truth, extra_info):
    value = ground_truth
    if isinstance(extra_info, Mapping):
        value = extra_info.get("target_objects", extra_info.get("objects", value))
    if isinstance(value, Mapping):
        value = value.get("target_objects", value.get("objects", value.get("ground_truth", "")))
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        objects = set()
        for item in value:
            objects.update(extract_coco_objects(item))
        return objects
    return extract_coco_objects(value)


def compute_score(solution_str, ground_truth, extra_info=None, normalization="strict", task=None, **kwargs):
    """Binary training outcome used by GRPO and causal interventions.

    The caption branch performs the Appendix-G COCO-category/synonym check. It
    is not the corpus-level official CHAIR-S/CHAIR-I evaluation metric.
    """
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError("outcome scorer requires a nonempty ground-truth answer")
    metadata = extra_info if isinstance(extra_info, dict) else {}
    task_name = normalize_text(task or metadata.get("task", ""), "strict")
    if task_name in {"caption", "chair"}:
        predicted_objects = extract_coco_objects(solution_str)
        expected_objects = _caption_ground_truth_objects(ground_truth, metadata)
        if not predicted_objects or not expected_objects:
            return 0.0
        return float(predicted_objects.issubset(expected_objects))

    predicted = normalize_text(extract_answer(solution_str), normalization)
    expected = normalize_text(extract_answer(ground_truth), normalization)
    if not predicted:
        return 0.0
    expected_binary = normalize_yes_no(expected)
    if expected_binary is not None:
        return float(normalize_yes_no(predicted) == expected_binary)
    return float(predicted == expected)
