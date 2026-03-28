
import sys
import json
import cv2
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_DIM               = 100    # minimum pixels per side
SCORE_PASS            = 45     # minimum score to accept

# Colour channel averages (overall image)
BLUE_HARD_MARGIN      = 30     # avg_b > avg_r + 30  → hard reject
GREEN_HARD_MARGIN     = 25     # avg_g > avg_r + 25  → hard reject

# ── KEY FIX 1: Bright-pixel saturation ───────────────────────────────────────
# Bright pixels = pixels with HSV Value > 80 (not dark background)
# For fundus: these pixels are the RETINA — highly saturated (S > 60)
# For screenshots / docs: these pixels are white/grey — S near 0
BRIGHT_PIX_SAT_REJECT = 35     # mean S of bright pixels < 35  → NOT a fundus
BRIGHT_PIX_SAT_BONUS  = 70     # mean S of bright pixels > 70  → likely fundus
BRIGHT_PIX_MIN_COUNT  = 0.05   # need at least 5% bright pixels to make this check

# ── KEY FIX 2: Bright-pixel warmth (R dominance in bright area) ──────────────
# Fundus: bright pixels have R >> B (orange/red retinal tissue)
# Screenshot: bright pixels have R ≈ B (white/grey)
BRIGHT_WARM_REJECT    = 5      # R-B of bright pixels < 5  → cold/neutral → reject
BRIGHT_WARM_BONUS     = 40     # R-B of bright pixels > 40 → warm → fundus bonus

# ── KEY FIX 3: Colourful-pixel coverage ──────────────────────────────────────
# At least N% of pixels must be meaningfully saturated
# This catches documents, X-rays, greyscale photos
COLOURFUL_COV_REJECT  = 0.05   # < 5% of pixels with S > 40  → almost no colour
COLOURFUL_COV_BONUS   = 0.25   # > 25% colourful pixels      → rich colour

# Edge density
EDGE_HARD_REJECT      = 0.15   # >15% edge pixels → screenshot/document
EDGE_SOFT_PENALTY     = 0.09   # >9%  edge pixels → suspicious
EDGE_SMOOTH_BONUS     = 0.04   # <4%  edge pixels → smooth

# Aspect ratio
ASPECT_MIN            = 0.4
ASPECT_MAX            = 2.5


def validate_fundus_image(image_path: str):
    """
    Returns (is_valid: bool, score: int 0–100, message: str)
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return False, 0, (
            "Cannot read the uploaded file. "
            "Please upload a valid JPEG or PNG image."
        )

    h, w = img.shape[:2]
    score = 50
    issues, bonuses = [], []

    # ── 1. Minimum size ───────────────────────────────────────────────────────
    if w < MIN_DIM or h < MIN_DIM:
        return False, 5, (
            f"Image is too small ({w}×{h} px). "
            f"Please upload at least {MIN_DIM}×{MIN_DIM} pixels."
        )

    # ── 2. Aspect ratio ───────────────────────────────────────────────────────
    aspect = w / h
    if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
        return False, 10, (
            "This image's aspect ratio is unusual for a retinal photograph "
            f"({aspect:.2f}). Fundus images are typically near-square. "
            "Please upload a proper fundus photograph."
        )

    # ── Convert to HSV once ───────────────────────────────────────────────────
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_chan = hsv[:, :, 1].astype(np.float32)   # 0–255
    v_chan = hsv[:, :, 2].astype(np.float32)   # 0–255

    # ── 3. Overall colour channel averages ────────────────────────────────────
    b_ch, g_ch, r_ch = cv2.split(img)
    avg_r = float(np.mean(r_ch))
    avg_g = float(np.mean(g_ch))
    avg_b = float(np.mean(b_ch))

    if avg_b > avg_r + BLUE_HARD_MARGIN:
        return False, 6, (
            "This does NOT appear to be a retinal fundus photograph.\n"
            f"The image is predominantly blue (avg R={avg_r:.0f}, B={avg_b:.0f}).\n"
            "Retinal fundus images are warm reddish-orange, never blue.\n"
            "Please upload a fundus photograph taken with ophthalmic equipment."
        )
    if avg_g > avg_r + GREEN_HARD_MARGIN:
        return False, 6, (
            "This does NOT appear to be a retinal fundus photograph.\n"
            f"The image is predominantly green (avg R={avg_r:.0f}, G={avg_g:.0f}).\n"
            "Retinal fundus images are warm reddish-orange, never green.\n"
            "Please upload a fundus photograph taken with ophthalmic equipment."
        )

    # Mild overall warmth bonus / penalty
    if avg_r > avg_b + 25:
        score += 10
    elif avg_r > avg_b + 8:
        score += 4
    elif avg_b > avg_r + 15:
        score -= 12
        issues.append(f"Cool colour tone (R={avg_r:.0f}, B={avg_b:.0f})")

    # ── 4. KEY FIX: Bright-pixel saturation ──────────────────────────────────
    bright_mask   = v_chan > 80
    bright_count  = int(bright_mask.sum())
    total_pixels  = h * w
    bright_frac   = bright_count / total_pixels

    if bright_frac >= BRIGHT_PIX_MIN_COUNT:
        bright_sat_mean = float(np.mean(s_chan[bright_mask]))

        if bright_sat_mean < BRIGHT_PIX_SAT_REJECT:
            # Bright pixels are WHITE / GREY — this is a document, screenshot, or X-ray
            return False, 8, (
                "This does NOT appear to be a retinal fundus photograph.\n"
                f"The bright areas of the image are white or grey "
                f"(colour saturation = {bright_sat_mean:.0f}/255).\n"
                "Retinal fundus images have coloured (orange/red) tissue in the bright areas.\n"
                "Documents, screenshots, X-rays, and tables are NOT accepted.\n"
                "Please upload a retinal fundus photograph taken with ophthalmic equipment."
            )

        if bright_sat_mean >= BRIGHT_PIX_SAT_BONUS:
            bonuses.append(f"Bright areas are richly coloured (S={bright_sat_mean:.0f}) — consistent with retinal tissue")
            score += 22
        elif bright_sat_mean >= 45:
            bonuses.append(f"Bright areas have reasonable colour (S={bright_sat_mean:.0f})")
            score += 10
        else:
            issues.append(f"Bright areas are weakly saturated (S={bright_sat_mean:.0f})")
            score -= 10

    # ── 5. KEY FIX: Bright-pixel warmth (R dominance in bright area) ──────────
    if bright_count > 100:
        r_bright = float(np.mean(r_ch.astype(np.float32)[bright_mask]))
        b_bright = float(np.mean(b_ch.astype(np.float32)[bright_mask]))
        warm_diff = r_bright - b_bright

        if warm_diff < BRIGHT_WARM_REJECT:
            # Bright areas are neutral white/grey → document / screenshot
            return False, 9, (
                "This does NOT appear to be a retinal fundus photograph.\n"
                f"The bright areas of the image are neutral white/grey "
                f"(R={r_bright:.0f}, B={b_bright:.0f}, difference={warm_diff:.0f}).\n"
                "In fundus photographs, the retinal tissue is warm reddish-orange.\n"
                "This image looks like a document, table, or screenshot.\n"
                "Please upload a retinal fundus photograph taken with ophthalmic equipment."
            )

        if warm_diff >= BRIGHT_WARM_BONUS:
            bonuses.append(f"Bright areas are warm reddish (R-B={warm_diff:.0f}) — matches retinal tissue colour")
            score += 18
        elif warm_diff >= 15:
            bonuses.append(f"Warm colour in bright areas (R-B={warm_diff:.0f})")
            score += 8

    # ── 6. Colourful pixel coverage ───────────────────────────────────────────
    # What fraction of ALL pixels have meaningful saturation?
    colourful_mask  = s_chan > 40
    colourful_frac  = float(colourful_mask.sum()) / total_pixels

    if colourful_frac < COLOURFUL_COV_REJECT:
        # Almost no colour anywhere — greyscale, X-ray, or solid colour image
        return False, 7, (
            "This does NOT appear to be a retinal fundus photograph.\n"
            f"Almost no coloured pixels were found ({colourful_frac*100:.1f}% of image).\n"
            "This is typical of greyscale photos, X-rays, or black-and-white documents.\n"
            "Retinal fundus photographs are colourful (warm orange/red).\n"
            "Please upload a fundus photograph taken with ophthalmic equipment."
        )

    if colourful_frac >= COLOURFUL_COV_BONUS:
        bonuses.append(f"Rich colour coverage ({colourful_frac*100:.0f}% of pixels coloured)")
        score += 12
    elif colourful_frac >= 0.10:
        score += 5

    # ── 7. Edge density ───────────────────────────────────────────────────────
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges       = cv2.Canny(gray, 80, 160)
    edge_ratio  = float(np.sum(edges > 0)) / total_pixels

    if edge_ratio > EDGE_HARD_REJECT:
        return False, 12, (
            "This does NOT appear to be a retinal fundus photograph.\n"
            f"Very high edge density ({edge_ratio*100:.1f}% of pixels are sharp edges).\n"
            "This is characteristic of documents, tables, buildings, or screenshots.\n"
            "Please upload a fundus photograph taken with ophthalmic equipment."
        )

    if edge_ratio > EDGE_SOFT_PENALTY:
        score -= 12
        issues.append(f"Moderate edge density ({edge_ratio*100:.1f}%)")
    elif edge_ratio < EDGE_SMOOTH_BONUS:
        score += 10
        bonuses.append(f"Smooth texture ({edge_ratio*100:.1f}% edges) — typical of fundus images")
    else:
        score += 3

    # ── 8. Nearly white / blank image ────────────────────────────────────────
    mean_val = float(np.mean(v_chan))
    mean_sat = float(np.mean(s_chan))

    if mean_val > 210 and mean_sat < 25:
        return False, 8, (
            "This image appears to be predominantly white/blank with very little colour.\n"
            "This is typical of blank documents, paper, or over-exposed images.\n"
            "Please upload a retinal fundus photograph taken with ophthalmic equipment."
        )

    if mean_val < 15:
        return False, 5, (
            "This image is almost completely black.\n"
            "Please upload a properly exposed retinal fundus photograph."
        )

    # ── Clamp & decide ────────────────────────────────────────────────────────
    score = max(0, min(100, score))

    if score < SCORE_PASS:
        problem_lines = "\n".join(f"  • {i}" for i in issues) if issues \
            else "  • Image characteristics do not match a retinal fundus photograph"
        return False, score, (
            f"This does NOT appear to be a retinal fundus photograph "
            f"(quality score: {score}/100).\n\n"
            f"Issues detected:\n{problem_lines}\n\n"
            "ACCEPTED: Retinal fundus photographs — circular, reddish-orange images "
            "showing the optic disc and blood vessels, taken with fundus camera equipment.\n"
            "NOT ACCEPTED: Screenshots, documents, tables, animals, buildings, "
            "faces, X-rays, MRI scans, or any non-retinal image."
        )

    detail = " | ".join(bonuses[:2]) if bonuses else "passed all checks"
    return True, score, f"Image accepted (quality score: {score}/100). {detail}."


# ── Self-test ──────────────────────────────────────────────────────────────────
def _self_test():
    """Quick sanity-check with synthetic images."""
    import os

    tests = []

    # Test 1: Dark navy + white table (the exact bug scenario)
    h, w = 600, 740
    navy_table = np.zeros((h, w, 3), dtype=np.uint8)
    navy_table[:, :]           = [61, 41, 25]    # BGR navy
    navy_table[150:450, 130:610] = [235, 235, 240]  # white/grey table
    path1 = '/tmp/test_navy_table.jpg'
    cv2.imwrite(path1, navy_table)
    valid, score, msg = validate_fundus_image(path1)
    tests.append(('Dark navy + white table (should REJECT)', valid, score, 'PASS' if not valid else 'FAIL'))

    # Test 2: Simulate a fundus image (orange circle on black)
    fundus_sim = np.zeros((600, 600, 3), dtype=np.uint8)
    cx, cy, r = 300, 300, 270
    Y, X = np.ogrid[:600, :600]
    dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
    mask = dist < r
    # Orange-red retinal tissue with some variation
    fundus_sim[mask, 2] = 170  # R (in BGR)
    fundus_sim[mask, 1] = 90   # G
    fundus_sim[mask, 0] = 50   # B
    # Add some noise to make it more realistic
    noise = np.random.randint(-20, 20, (600, 600, 3))
    fundus_sim = np.clip(fundus_sim.astype(int) + noise, 0, 255).astype(np.uint8)
    path2 = '/tmp/test_fundus_sim.jpg'
    cv2.imwrite(path2, fundus_sim)
    valid2, score2, msg2 = validate_fundus_image(path2)
    tests.append(('Simulated fundus (should ACCEPT)', valid2, score2, 'PASS' if valid2 else 'FAIL'))

    # Test 3: White/grey document
    doc = np.ones((800, 600, 3), dtype=np.uint8) * 240
    path3 = '/tmp/test_doc.jpg'
    cv2.imwrite(path3, doc)
    valid3, score3, msg3 = validate_fundus_image(path3)
    tests.append(('White document (should REJECT)', valid3, score3, 'PASS' if not valid3 else 'FAIL'))

    print("\n=== SELF-TEST RESULTS ===")
    all_pass = True
    for name, v, s, result in tests:
        print(f"  {result}  |  {name}  |  valid={v}, score={s}")
        if result == 'FAIL':
            all_pass = False

    print(f"\n{'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    return all_pass


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        success = _self_test()
        sys.exit(0 if success else 1)

    if len(sys.argv) != 2:
        print(json.dumps({
            'valid': False, 'score': 0,
            'message': 'Usage: python validate_image.py <image_path>'
        }))
        sys.exit(1)

    valid, score, message = validate_fundus_image(sys.argv[1])
    print(json.dumps({'valid': valid, 'score': score, 'message': message}))
    sys.exit(0 if valid else 1)