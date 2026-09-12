import json


def write_report(path, report):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def update_distribution(distributions, tag_name, tag):
    value = tag.get("value")
    key = json.dumps(value, ensure_ascii=False, sort_keys=True)
    distributions.setdefault(tag_name, {})
    distributions[tag_name][key] = distributions[tag_name].get(key, 0) + 1
