import json
import os
import os.path as osp
import re
from collections import defaultdict

from datasets import load_dataset


PROMPT = (
    "Please convert this document image to Markdown format. "
    "For tables, output HTML table tags. "
    "For mathematical formulas, output LaTeX. "
    "For plain text, output the text directly. "
    "Maintain the reading order of the document."
)


def _text_norm(text):
    text = re.sub(r"_{4,}", "____", text)
    text = re.sub(r" {4,}", "    ", text)
    text = re.sub(r"([^a-zA-Z0-9])\1{10,}", r"\1\1\1\1", text)
    return text.replace("/t", "\t").replace("/n", "\n")


def _build_gt_markdown(sample):
    """Reconstruct ground-truth Markdown from layout_dets, following
    the logic in OmniDocBench's tools/json2md.py."""
    annos = [x for x in sample["layout_dets"] if x.get("order") is not None]

    # Handle truncated text blocks — merge them by reading order.
    truncated_ids = {}
    related_groups = []
    for relation in sample.get("extra", {}).get("relation", []):
        if relation.get("relation_type") == "truncated":
            src, tgt = relation["source_anno_id"], relation["target_anno_id"]
            truncated_ids[src] = ""
            truncated_ids[tgt] = ""
            merged = False
            for group in related_groups:
                if src in group or tgt in group:
                    group.add(src)
                    group.add(tgt)
                    merged = True
                    break
            if not merged:
                related_groups.append({src, tgt})

    anno_by_id = {}
    merged_annos = []
    for item in annos:
        if item["anno_id"] in truncated_ids:
            anno_by_id[item["anno_id"]] = item
        else:
            merged_annos.append(item)

    for group in related_groups:
        blocks = [anno_by_id[aid] for aid in group if aid in anno_by_id]
        if not blocks:
            continue
        blocks.sort(key=lambda x: x["order"])
        text = ""
        for block in blocks:
            content = block.get("text", "")
            if not content:
                continue
            if content.endswith("-"):
                text += content[:-1]
            elif text and text[-1] != " ":
                text += " " + content
            else:
                text += content
        merged_annos.append(
            {
                "category_type": blocks[0]["category_type"],
                "order": blocks[0]["order"],
                "anno_id": blocks[0]["anno_id"],
                "text": text,
                "html": blocks[0].get("html", ""),
                "latex": blocks[0].get("latex", ""),
            }
        )

    annos = sorted(merged_annos, key=lambda x: x["order"])

    parts = []
    for anno in annos:
        cat = anno["category_type"]

        if cat == "table":
            html_content = anno.get("html", "")
            if html_content:
                parts.append(html_content)
            continue

        if cat == "title":
            text = anno.get("text", "").strip().strip("#").strip()
            if text:
                parts.append("# " + _text_norm(text))
            continue

        if cat in ("equation_isolated", "display_formula"):
            latex = anno.get("latex", "")
            if latex:
                parts.append("$$\n" + latex.strip() + "\n$$")
            continue

        if cat == "figure":
            parts.append("![figure]()")
            continue

        # Default: text_block, header, footer, page_number, code, etc.
        text = anno.get("text", "")
        if text:
            parts.append(_text_norm(text))
        elif anno.get("latex"):
            parts.append(anno["latex"])
        elif anno.get("html"):
            parts.append(anno["html"])

    return "\n\n".join(parts)


def _get_page_attribute(sample, key, default="unknown"):
    return (
        sample.get("page_info", {})
        .get("page_attribute", {})
        .get(key, default)
    )


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "img"), exist_ok=True)

    dataset = load_dataset(data_dir, split=split)

    content = []
    for i, sample in enumerate(dataset):
        image_path = sample.get("page_info", {}).get("image_path", f"{i}.png")
        base_name = osp.splitext(osp.basename(image_path))[0]
        question_id = base_name

        gt_markdown = _build_gt_markdown(sample)

        data_source = _get_page_attribute(sample, "data_source", "unknown")
        language = _get_page_attribute(sample, "language", "unknown")
        subset = _get_page_attribute(sample, "subset", "v1.5")

        info = {
            "question_id": question_id,
            "question": f"<image 1>{PROMPT}",
            "answer": gt_markdown,
            "question_type": "doc-parsing",
            "sub_task": data_source,
            "language": language,
            "subset": subset,
            "img_path": f"img/{base_name}.png",
        }

        img = sample.get("image")
        if img is not None:
            img_save_path = osp.join(output_dir, f"img/{base_name}.png")
            os.makedirs(osp.dirname(img_save_path), exist_ok=True)
            try:
                img.save(img_save_path)
            except Exception as e:
                print(f"Error saving image {question_id}: {e}")
                continue

        content.append(info)

    with open(osp.join(output_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} pages. Data saved to {osp.join(output_dir, 'data.json')}")
