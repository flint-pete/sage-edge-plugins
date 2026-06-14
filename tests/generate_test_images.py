"""
Generate synthetic test images for Sage plugin testing.

Creates three images simulating typical Sage node camera captures:
  1. urban_street.jpg  — Urban scene with rectangles representing cars/people
  2. wildlife.jpg      — Nature scene with shapes representing birds/animals
  3. sky_clouds.jpg    — Sky scene for cloud cover / general description

These are deliberately simple synthetic images. They won't produce
meaningful ML results but they exercise the full plugin pipeline:
camera capture -> preprocessing -> model inference -> publish.
"""
import os
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(__file__), "sample-images")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


# ── 1. Urban street scene (640x480) ────────────────────────────────
def make_urban():
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Sky gradient (top half)
    for y in range(200):
        blue = int(180 + 75 * (1 - y / 200))
        img[y, :] = (blue, 150, 80)

    # Road (bottom)
    img[300:, :] = (60, 60, 60)

    # Sidewalk
    img[280:300, :] = (160, 160, 160)

    # Buildings (gray rectangles)
    cv2.rectangle(img, (20, 80), (120, 280), (130, 130, 140), -1)
    cv2.rectangle(img, (20, 80), (120, 280), (90, 90, 100), 2)
    cv2.rectangle(img, (150, 120), (250, 280), (120, 120, 130), -1)
    cv2.rectangle(img, (150, 120), (250, 280), (80, 80, 90), 2)
    cv2.rectangle(img, (420, 100), (550, 280), (140, 140, 150), -1)
    cv2.rectangle(img, (420, 100), (550, 280), (100, 100, 110), 2)

    # Windows on buildings
    for bx, by, bw, bh in [(20, 80, 100, 200), (150, 120, 100, 160), (420, 100, 130, 180)]:
        for wy in range(by + 15, by + bh - 10, 30):
            for wx in range(bx + 10, bx + bw - 10, 25):
                cv2.rectangle(img, (wx, wy), (wx + 15, wy + 20), (180, 200, 220), -1)

    # Cars (colored rectangles on road)
    cv2.rectangle(img, (100, 340), (200, 400), (0, 0, 200), -1)    # red car
    cv2.rectangle(img, (100, 340), (200, 400), (0, 0, 150), 2)
    cv2.rectangle(img, (300, 350), (420, 420), (200, 200, 200), -1) # silver car
    cv2.rectangle(img, (300, 350), (420, 420), (150, 150, 150), 2)
    cv2.rectangle(img, (500, 330), (600, 390), (200, 100, 0), -1)   # blue car
    cv2.rectangle(img, (500, 330), (600, 390), (150, 80, 0), 2)

    # People (stick figures on sidewalk)
    for px in [260, 340, 380]:
        # Head
        cv2.circle(img, (px, 258), 8, (180, 140, 120), -1)
        # Body
        cv2.line(img, (px, 266), (px, 290), (100, 80, 60), 2)
        # Legs
        cv2.line(img, (px, 290), (px - 5, 300), (100, 80, 60), 2)
        cv2.line(img, (px, 290), (px + 5, 300), (100, 80, 60), 2)

    # Lane markings
    for x in range(50, 640, 80):
        cv2.rectangle(img, (x, 370), (x + 40, 375), (255, 255, 255), -1)

    draw_text(img, "TEST: urban_street.jpg", (10, 470), 0.5)
    path = os.path.join(OUT_DIR, "urban_street.jpg")
    cv2.imwrite(path, img)
    print(f"Created {path} ({img.shape})")
    return path


# ── 2. Wildlife / nature scene (640x480) ───────────────────────────
def make_wildlife():
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Sky
    for y in range(180):
        img[y, :] = (int(200 + 55 * (1 - y / 180)), 160, 100)

    # Grass / ground
    for y in range(180, 480):
        green = int(80 + 60 * np.sin((y - 180) * 0.05))
        img[y, :] = (40, green, 30)

    # Trees
    for tx, th in [(80, 200), (200, 180), (500, 220)]:
        # Trunk
        cv2.rectangle(img, (tx - 8, 180), (tx + 8, 180 + th // 2), (40, 60, 100), -1)
        # Canopy
        cv2.circle(img, (tx, 180 - th // 4), th // 3, (30, int(100 + np.random.randint(60)), 20), -1)

    # Birds (V shapes in sky)
    for bx, by in [(100, 60), (200, 40), (350, 70), (450, 50), (520, 80)]:
        cv2.line(img, (bx - 10, by + 5), (bx, by), (20, 20, 20), 2)
        cv2.line(img, (bx, by), (bx + 10, by + 5), (20, 20, 20), 2)

    # Deer-like shape (brown ellipse + legs)
    cx, cy = 350, 320
    cv2.ellipse(img, (cx, cy), (50, 25), 0, 0, 360, (50, 80, 140), -1)
    # Head
    cv2.circle(img, (cx + 55, cy - 15), 12, (50, 80, 140), -1)
    # Legs
    for lx in [cx - 30, cx - 15, cx + 15, cx + 30]:
        cv2.line(img, (lx, cy + 20), (lx, cy + 55), (40, 70, 120), 3)
    # Antlers
    cv2.line(img, (cx + 55, cy - 27), (cx + 45, cy - 50), (40, 70, 120), 2)
    cv2.line(img, (cx + 55, cy - 27), (cx + 65, cy - 50), (40, 70, 120), 2)

    # Rabbit-like shape (small ellipse)
    rx, ry = 180, 380
    cv2.ellipse(img, (rx, ry), (15, 10), 0, 0, 360, (140, 160, 180), -1)
    cv2.circle(img, (rx + 15, ry - 5), 6, (140, 160, 180), -1)
    cv2.ellipse(img, (rx + 18, ry - 15), (3, 8), 10, 0, 360, (140, 160, 180), -1)

    # Pond
    cv2.ellipse(img, (450, 400), (70, 30), 0, 0, 360, (150, 100, 50), -1)

    draw_text(img, "TEST: wildlife.jpg", (10, 470), 0.5)
    path = os.path.join(OUT_DIR, "wildlife.jpg")
    cv2.imwrite(path, img)
    print(f"Created {path} ({img.shape})")
    return path


# ── 3. Sky / clouds scene (640x480) ────────────────────────────────
def make_sky():
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Deep blue sky gradient
    for y in range(480):
        blue = int(240 - 80 * (y / 480))
        green = int(180 - 40 * (y / 480))
        img[y, :] = (blue, green, 100 + int(30 * y / 480))

    # Clouds (white/gray ellipses)
    rng = np.random.RandomState(42)
    for _ in range(8):
        cx = rng.randint(50, 590)
        cy = rng.randint(30, 350)
        rx = rng.randint(40, 120)
        ry = rng.randint(20, 50)
        brightness = rng.randint(200, 255)
        cv2.ellipse(img, (cx, cy), (rx, ry), rng.randint(-20, 20),
                    0, 360, (brightness, brightness, brightness), -1)
        # Secondary puffs
        for _ in range(3):
            dx = rng.randint(-rx, rx)
            dy = rng.randint(-ry // 2, ry // 2)
            sr = rng.randint(15, rx // 2)
            b2 = rng.randint(190, brightness)
            cv2.ellipse(img, (cx + dx, cy + dy), (sr, sr // 2),
                        rng.randint(-10, 10), 0, 360, (b2, b2, b2), -1)

    # Sun
    cv2.circle(img, (530, 70), 40, (100, 220, 255), -1)
    # Sun glow
    for r in range(60, 100, 10):
        alpha = 1.0 - (r - 60) / 40
        overlay = img.copy()
        cv2.circle(overlay, (530, 70), r, (100, 200, 240), -1)
        cv2.addWeighted(overlay, alpha * 0.15, img, 1 - alpha * 0.15, 0, img)

    # Thin horizon line with ground
    img[440:, :] = (50, 80, 40)
    cv2.line(img, (0, 440), (640, 440), (80, 100, 60), 2)

    draw_text(img, "TEST: sky_clouds.jpg", (10, 470), 0.5)
    path = os.path.join(OUT_DIR, "sky_clouds.jpg")
    cv2.imwrite(path, img)
    print(f"Created {path} ({img.shape})")
    return path


if __name__ == "__main__":
    make_urban()
    make_wildlife()
    make_sky()
    print("\nDone — 3 test images generated in", OUT_DIR)
