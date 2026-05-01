"""Image annotation helpers for jpy-ml."""

_jpy_ann_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 0, 255),
    (0, 255, 255), (255, 192, 203), (255, 255, 0), (128, 0, 255), (0, 128, 128)
]


def jpy_annotate(input_path, output_path, boxes, task):
    """Annotate an image with detection/segmentation/pose/OBB results.

    Args:
        input_path: Path to source image.
        output_path: Where to save annotated image.
        boxes: List of dicts with box/keypoint/polygon data.
        task: Task type string (detect, segment, classify, pose, obb).

    Returns:
        output_path on success.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import math

        img = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Try to get a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            except (OSError, IOError):
                font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            color = _jpy_ann_colors[i % len(_jpy_ann_colors)]

            if task == "detect" or task == "segment":
                if "x1" in box:
                    draw.rectangle(
                        [box["x1"], box["y1"], box["x2"], box["y2"]],
                        outline=color, width=2
                    )
                label = f"{box.get('class_name', '?')} {box.get('confidence', 0):.2f}"
                draw.text((box.get("x1", 0), box.get("y1", 0) - 16), label, fill=color, font=font)

                if task == "segment" and "polygon" in box:
                    pts = [(p[0], p[1]) for p in box["polygon"]]
                    if pts:
                        draw.polygon(pts, outline=color, fill=(*color[:3], 40))

            elif task == "classify":
                label = f"{box.get('class_name', '?')} {box.get('confidence', 0):.2f}"
                draw.text((10, 10 + i * 20), label, fill=color, font=font)

            elif task == "pose":
                if "x1" in box:
                    draw.rectangle(
                        [box["x1"], box["y1"], box["x2"], box["y2"]],
                        outline=color, width=2
                    )
                if "keypoints" in box:
                    for kpt in box["keypoints"]:
                        x, y, c = kpt[0], kpt[1], kpt[2]
                        if c > 0.5:
                            r = 3
                            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

            elif task == "obb":
                cx, cy = box.get("cx", 0), box.get("cy", 0)
                w, h = box.get("w", 0), box.get("h", 0)
                angle = box.get("angle", 0)
                cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                corners = [
                    (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
                    for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
                ]
                draw.polygon(corners, outline=color, width=2)
                label = f"{box.get('class_name', '?')} {box.get('confidence', 0):.2f}"
                draw.text((cx, cy - 16), label, fill=color, font=font)

        img.save(output_path)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Annotation failed: {e}") from e
