#!/usr/bin/env python3
"""
Generate ECR icon (512x512) and science image (1920x1080) for each plugin.

Each plugin gets a distinctive color scheme and visual motif:
  - YOLO Object Counter:        deep blue/cyan — bounding boxes grid
  - BioCLIP Species Classifier: green/emerald — taxonomic tree / leaf
  - vLLM Edge Inference:        purple/violet — neural network / brain

All images are clean, professional, and informative.
"""
import math
import os
import random
from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGINS = {
    "yolo-object-counter": {
        "title": "YOLO",
        "subtitle": "Object Counter",
        "tagline": "Real-time object detection & counting at the edge",
        "color_primary": (0, 120, 200),      # deep blue
        "color_secondary": (0, 200, 220),     # cyan
        "color_accent": (255, 200, 0),        # amber
        "color_dark": (10, 25, 50),           # navy
        "icon_symbol": "bbox",                # bounding box motif
        "science_details": [
            "Model: YOLOv11x (57M params)",
            "80 COCO classes • 32ms/frame @ 4K",
            "Per-class counting with NMS filtering",
            "Annotated image upload via pywaggle",
        ],
    },
    "bioclip-species-classifier": {
        "title": "BioCLIP",
        "subtitle": "Species Classifier",
        "tagline": "Taxonomic classification from Kingdom to Species",
        "color_primary": (16, 130, 70),       # forest green
        "color_secondary": (40, 180, 100),    # emerald
        "color_accent": (200, 230, 80),       # lime
        "color_dark": (8, 35, 20),            # deep forest
        "icon_symbol": "leaf",                # leaf/taxonomy motif
        "science_details": [
            "Model: BioCLIP-2 (430M params)",
            "TreeOfLife-200M embeddings",
            "7 taxonomic ranks: Kingdom → Species",
            "Zero-shot biological classification",
        ],
    },
    "vllm-edge-inference": {
        "title": "vLLM",
        "subtitle": "Edge Inference",
        "tagline": "Vision-language scene understanding at the edge",
        "color_primary": (100, 40, 180),      # deep purple
        "color_secondary": (160, 80, 220),    # violet
        "color_accent": (255, 120, 200),      # pink
        "color_dark": (20, 10, 40),           # midnight
        "icon_symbol": "brain",               # neural/brain motif
        "science_details": [
            "Model: Qwen3-VL-32B-Instruct",
            "67GB model • 3.8 tok/s on GB10",
            "Natural language scene descriptions",
            "OpenAI-compatible vLLM serving",
        ],
    },
}

# ── Fonts ────────────────────────────────────────────────────────────
def load_font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_mono_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ── Drawing helpers ──────────────────────────────────────────────────
def blend(c1, c2, t):
    """Blend two colors: t=0 gives c1, t=1 gives c2."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_gradient_rect(draw, bbox, c1, c2, vertical=True):
    """Fill a rectangle with a gradient."""
    x1, y1, x2, y2 = bbox
    if vertical:
        for y in range(y1, y2):
            t = (y - y1) / max(y2 - y1, 1)
            draw.line([(x1, y), (x2, y)], fill=blend(c1, c2, t))
    else:
        for x in range(x1, x2):
            t = (x - x1) / max(x2 - x1, 1)
            draw.line([(x, y1), (x, y2)], fill=blend(c1, c2, t))


def draw_rounded_rect(draw, bbox, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = bbox
    draw.rounded_rectangle(bbox, radius=radius, fill=fill, outline=outline, width=width)


# ── YOLO Icon: bounding box grid ────────────────────────────────────
def draw_yolo_icon_motif(draw, cx, cy, size, colors):
    """Draw stylized bounding boxes — the essence of object detection."""
    s = size * 0.35

    # Several overlapping bounding boxes at different positions
    boxes = [
        (-0.4, -0.3, 0.1, 0.2),   # top-left box
        (0.0, -0.1, 0.5, 0.5),    # center-right box
        (-0.2, 0.1, 0.3, 0.45),   # bottom-center box
    ]
    box_colors = [colors["accent"], colors["secondary"], (255, 255, 255)]

    for i, (bx1, by1, bx2, by2) in enumerate(boxes):
        x1 = int(cx + bx1 * s * 2)
        y1 = int(cy + by1 * s * 2)
        x2 = int(cx + bx2 * s * 2)
        y2 = int(cy + by2 * s * 2)
        lw = max(3, int(size * 0.015))

        # Draw bounding box with corner emphasis
        c = box_colors[i]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=lw)

        # Corner markers (thicker, shorter)
        corner_len = int((x2 - x1) * 0.2)
        clw = lw + 2
        # top-left
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=c, width=clw)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=c, width=clw)
        # top-right
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=c, width=clw)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=c, width=clw)
        # bottom-left
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=c, width=clw)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=c, width=clw)
        # bottom-right
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=c, width=clw)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=c, width=clw)

    # Small crosshair in center
    ch = int(size * 0.04)
    draw.line([(cx - ch, cy), (cx + ch, cy)], fill=colors["accent"], width=2)
    draw.line([(cx, cy - ch), (cx, cy + ch)], fill=colors["accent"], width=2)


# ── BioCLIP Icon: leaf/DNA ──────────────────────────────────────────
def draw_bioclip_icon_motif(draw, cx, cy, size, colors):
    """Draw a stylized leaf with veins — representing biological classification."""
    s = size * 0.3

    # Leaf shape using ellipses (rotated via polygon)
    leaf_points = []
    for i in range(60):
        angle = (i / 60) * 2 * math.pi
        # Leaf-shaped parametric curve
        r = s * (0.8 + 0.2 * math.cos(2 * angle)) * (0.5 + 0.5 * math.sin(angle))
        x = cx + r * math.cos(angle) * 0.7
        y = cy - r * math.sin(angle) * 1.0 + s * 0.15
        leaf_points.append((x, y))

    draw.polygon(leaf_points, fill=colors["secondary"], outline=colors["accent"], width=2)

    # Central vein
    draw.line([(cx, cy - s * 0.7), (cx, cy + s * 0.65)], fill=colors["dark"], width=3)

    # Side veins
    for i in range(4):
        t = 0.15 + i * 0.18
        vy = int(cy - s * 0.7 + (s * 1.35) * t)
        vlen = int(s * 0.3 * (1 - abs(t - 0.5)))
        draw.line([(cx, vy), (cx + vlen, vy - vlen // 2)], fill=colors["dark"], width=2)
        draw.line([(cx, vy), (cx - vlen, vy - vlen // 2)], fill=colors["dark"], width=2)

    # Small DNA helix below leaf (taxonomy)
    for i in range(8):
        t = i / 7
        x_off = math.sin(t * math.pi * 2) * s * 0.15
        y_pos = cy + s * 0.5 + t * s * 0.35
        r = 3
        draw.ellipse([cx + x_off - r, y_pos - r, cx + x_off + r, y_pos + r],
                     fill=colors["accent"])
        draw.ellipse([cx - x_off - r, y_pos - r, cx - x_off + r, y_pos + r],
                     fill=(255, 255, 255))


# ── vLLM Icon: brain/network ────────────────────────────────────────
def draw_vllm_icon_motif(draw, cx, cy, size, colors):
    """Draw a stylized neural network / brain — representing LLM inference."""
    s = size * 0.3
    random.seed(42)

    # Neural network layers
    layers = [3, 5, 5, 3]
    layer_x = [cx - s * 0.6, cx - s * 0.2, cx + s * 0.2, cx + s * 0.6]
    layer_nodes = []

    for li, (num, lx) in enumerate(zip(layers, layer_x)):
        nodes = []
        for ni in range(num):
            ny = cy + (ni - (num - 1) / 2) * s * 0.3
            nodes.append((lx, ny))
        layer_nodes.append(nodes)

    # Draw connections
    for li in range(len(layers) - 1):
        for n1 in layer_nodes[li]:
            for n2 in layer_nodes[li + 1]:
                alpha = random.randint(60, 150)
                c = (*colors["secondary"][:3],)
                draw.line([n1, n2], fill=c, width=1)

    # Draw nodes
    for li, nodes in enumerate(layer_nodes):
        r = int(s * 0.06)
        for nx, ny in nodes:
            if li == 0:
                c = colors["accent"]
            elif li == len(layers) - 1:
                c = (255, 255, 255)
            else:
                c = colors["secondary"]
            draw.ellipse([nx - r, ny - r, nx + r, ny + r], fill=c, outline=colors["primary"], width=2)

    # Eye symbol (vision) — small
    ey = cy - s * 0.55
    er = int(s * 0.12)
    # Eye outline
    eye_pts = []
    for i in range(30):
        t = i / 29
        angle = -math.pi * 0.3 + t * math.pi * 0.6
        x = cx + er * 2 * math.cos(angle)
        y = ey + er * math.sin(angle)
        eye_pts.append((x, y))
    for i in range(30):
        t = i / 29
        angle = math.pi * 0.3 + math.pi - t * math.pi * 0.6
        x = cx + er * 2 * math.cos(angle)
        y = ey + er * math.sin(angle)
        eye_pts.append((x, y))
    draw.polygon(eye_pts, fill=colors["accent"], outline=colors["primary"], width=2)
    # Pupil
    pr = int(er * 0.4)
    draw.ellipse([cx - pr, ey - pr, cx + pr, ey + pr], fill=colors["dark"])


# ── Generate Icon (512x512) ─────────────────────────────────────────
def generate_icon(plugin_name, config):
    size = 512
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    c = config
    colors = {
        "primary": c["color_primary"],
        "secondary": c["color_secondary"],
        "accent": c["color_accent"],
        "dark": c["color_dark"],
    }

    # Background gradient
    draw_gradient_rect(draw, [0, 0, size, size], c["color_dark"], blend(c["color_dark"], c["color_primary"], 0.3))

    # Subtle grid pattern
    for i in range(0, size, 32):
        draw.line([(i, 0), (i, size)], fill=blend(c["color_dark"], c["color_primary"], 0.15), width=1)
        draw.line([(0, i), (size, i)], fill=blend(c["color_dark"], c["color_primary"], 0.15), width=1)

    # Draw the motif
    motif_funcs = {
        "bbox": draw_yolo_icon_motif,
        "leaf": draw_bioclip_icon_motif,
        "brain": draw_vllm_icon_motif,
    }
    motif_funcs[c["icon_symbol"]](draw, size // 2, size // 2 - 30, size, colors)

    # Title text
    title_font = load_font(48, bold=True)
    sub_font = load_font(24, bold=False)

    title = c["title"]
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw = tb[2] - tb[0]
    draw.text(((size - tw) // 2, size - 120), title, fill=(255, 255, 255), font=title_font)

    sub = c["subtitle"]
    sb = draw.textbbox((0, 0), sub, font=sub_font)
    sw = sb[2] - sb[0]
    draw.text(((size - sw) // 2, size - 65), sub, fill=c["color_secondary"], font=sub_font)

    # Sage badge — bottom corner
    badge_font = load_font(14, bold=True)
    draw.rounded_rectangle([size - 100, size - 30, size - 8, size - 8],
                           radius=4, fill=c["color_primary"])
    draw.text((size - 90, size - 28), "SAGE / ECR", fill=(255, 255, 255), font=badge_font)

    # Border
    draw.rounded_rectangle([2, 2, size - 3, size - 3], radius=16,
                           outline=c["color_primary"], width=3)

    out_path = os.path.join(BASE, "plugins", plugin_name, "ecr-meta", "ecr-icon.jpg")
    img.save(out_path, "JPEG", quality=95)
    print(f"  ✓ Icon: {out_path}")
    return out_path


# ── Generate Science Image (1920x1080) ──────────────────────────────
def generate_science_image(plugin_name, config):
    W, H = 1920, 1080
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    c = config
    colors = {
        "primary": c["color_primary"],
        "secondary": c["color_secondary"],
        "accent": c["color_accent"],
        "dark": c["color_dark"],
    }

    # Background gradient
    draw_gradient_rect(draw, [0, 0, W, H], c["color_dark"], blend(c["color_dark"], c["color_primary"], 0.25))

    # Grid pattern
    for i in range(0, max(W, H), 48):
        if i < W:
            draw.line([(i, 0), (i, H)], fill=blend(c["color_dark"], c["color_primary"], 0.12), width=1)
        if i < H:
            draw.line([(0, i), (W, i)], fill=blend(c["color_dark"], c["color_primary"], 0.12), width=1)

    # === Left panel: motif + title ===

    # Draw motif (large, centered in left half)
    motif_funcs = {
        "bbox": draw_yolo_icon_motif,
        "leaf": draw_bioclip_icon_motif,
        "brain": draw_vllm_icon_motif,
    }
    motif_funcs[c["icon_symbol"]](draw, W // 4, H // 2 - 80, 600, colors)

    # Title
    title_font = load_font(72, bold=True)
    sub_font = load_font(36, bold=False)
    tag_font = load_font(28, bold=False)

    title = c["title"]
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw = tb[2] - tb[0]
    draw.text(((W // 2 - tw) // 2, H - 250), title, fill=(255, 255, 255), font=title_font)

    sub = c["subtitle"]
    sb = draw.textbbox((0, 0), sub, font=sub_font)
    sw = sb[2] - sb[0]
    draw.text(((W // 2 - sw) // 2, H - 170), sub, fill=c["color_secondary"], font=sub_font)

    tag = c["tagline"]
    tgb = draw.textbbox((0, 0), tag, font=tag_font)
    tgw = tgb[2] - tgb[0]
    draw.text(((W // 2 - tgw) // 2, H - 120), tag, fill=blend(c["color_secondary"], (255, 255, 255), 0.5), font=tag_font)

    # === Right panel: info card ===
    card_x = W // 2 + 40
    card_y = 60
    card_w = W - card_x - 60
    card_h = H - 120
    card_bg = blend(c["color_dark"], c["color_primary"], 0.1)

    # Card background
    draw_rounded_rect(draw, [card_x, card_y, card_x + card_w, card_y + card_h],
                      radius=16, fill=(*card_bg, ),
                      outline=c["color_primary"], width=2)

    # Card header
    header_font = load_font(32, bold=True)
    draw.text((card_x + 30, card_y + 25), "Plugin Specifications", fill=c["color_accent"], font=header_font)

    # Divider
    draw.line([(card_x + 30, card_y + 75), (card_x + card_w - 30, card_y + 75)],
              fill=c["color_primary"], width=2)

    # Specs
    detail_font = load_font(24, bold=False)
    mono_font = load_mono_font(22)
    y = card_y + 95

    for detail in c["science_details"]:
        # Bullet
        draw.ellipse([card_x + 35, y + 8, card_x + 45, y + 18], fill=c["color_accent"])
        draw.text((card_x + 55, y), detail, fill=(220, 220, 230), font=detail_font)
        y += 45

    # Architecture section
    y += 20
    draw.text((card_x + 30, y), "Architecture", fill=c["color_accent"], font=header_font)
    y += 45
    draw.line([(card_x + 30, y), (card_x + card_w - 30, y)], fill=c["color_primary"], width=1)
    y += 15

    arch_items = [
        "Platform: Sage Continuum / Waggle",
        "Target: NVIDIA GB10 (128GB unified)",
        "Runtime: Docker + pywaggle SDK",
        "Architecture: linux/arm64, linux/amd64",
    ]
    for item in arch_items:
        draw.text((card_x + 40, y), "▸", fill=c["color_secondary"], font=detail_font)
        draw.text((card_x + 60, y), item, fill=(200, 200, 210), font=detail_font)
        y += 38

    # Data pipeline section
    y += 20
    draw.text((card_x + 30, y), "Data Pipeline", fill=c["color_accent"], font=header_font)
    y += 45
    draw.line([(card_x + 30, y), (card_x + card_w - 30, y)], fill=c["color_primary"], width=1)
    y += 15

    pipe_steps = ["Camera", "Inference", "Publish", "Beehive"]
    pipe_x = card_x + 40
    step_font = load_font(22, bold=True)

    # Draw pipeline as connected boxes
    box_y = y + 5
    box_h = 40
    step_x = card_x + 40
    for i, step_text in enumerate(pipe_steps):
        stb = draw.textbbox((0, 0), step_text, font=step_font)
        stw = stb[2] - stb[0] + 24

        draw_rounded_rect(draw, [step_x, box_y, step_x + stw, box_y + box_h],
                          radius=6, fill=c["color_primary"],
                          outline=c["color_secondary"], width=1)
        draw.text((step_x + 12, box_y + 8), step_text, fill=(255, 255, 255), font=step_font)

        if i < len(pipe_steps) - 1:
            arrow_x = step_x + stw + 8
            draw.text((arrow_x, box_y + 6), "→", fill=c["color_accent"], font=step_font)
            step_x = arrow_x + 28
        else:
            step_x = step_x + stw + 5

    # Footer
    footer_font = load_font(18, bold=False)
    footer_y = H - 50
    draw.text((40, footer_y), "NSF Award #1935984  •  Sage Continuum  •  sagecontinuum.org",
              fill=blend(c["color_primary"], (255, 255, 255), 0.5), font=footer_font)

    sage_badge = load_font(20, bold=True)
    draw.rounded_rectangle([W - 180, footer_y - 5, W - 20, footer_y + 25],
                           radius=4, fill=c["color_primary"])
    draw.text((W - 168, footer_y - 2), "SAGE / ECR", fill=(255, 255, 255), font=sage_badge)

    out_path = os.path.join(BASE, "plugins", plugin_name, "ecr-meta", "ecr-science-image.jpg")
    img.save(out_path, "JPEG", quality=95)
    print(f"  ✓ Science image: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating ECR images for all plugins...\n")

    for name, config in PLUGINS.items():
        print(f"[{config['title']} {config['subtitle']}]")
        generate_icon(name, config)
        generate_science_image(name, config)
        print()

    print("Done! All 6 images generated.")
