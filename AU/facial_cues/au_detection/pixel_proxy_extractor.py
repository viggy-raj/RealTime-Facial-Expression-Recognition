import cv2
import numpy as np


class PixelProxyExtractor:
    """
    Region-based AU proxy extractor for 48x48 grayscale CK+ face images.

    Divides the face into 9 anatomically-motivated regions and computes
    gradient-based descriptors (mean gradient magnitude, vertical/horizontal
    gradient ratios, brightness) per region to produce 18 discriminative
    pseudo-AU intensities.

    This replaces the original 2-statistic (eye_var / mouth_var) approach
    which could not distinguish between 6+ emotion classes.  The new
    descriptor produces meaningful spatial structure per emotion:
      - Disgust:  high nose/brow gradients
      - Happy:    high cheek + lip-corner activity
      - Fear:     high brow + wide-eye signal
      - Surprise: very high brow + open-mouth
      - Contempt: asymmetric lip signal
      - Sad:      low-intensity drooped brow
    """

    # Standard 18 AU names used in the pipeline
    AU_NAMES = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26",
        "AU28", "AU45",
    ]

    def __init__(self):
        self.model = True  # flag expected by orchestrator

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, img_bgr, bbox=None, *args, **kwargs):
        """
        img_bgr: BGR image (webcam frame or dataset frame).
        bbox:    (x, y, w, h) face bounding box.  If None, uses entire image.
        Returns: dict  {AU_NAME -> float in [0, 1]}
        """
        face = self._crop_face(img_bgr, bbox)
        if face.size == 0:
            return {au: 0.0 for au in self.AU_NAMES}

        # Normalise to 48×48 float32 [0, 1]  (matches CK+48 training domain)
        gray = (cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                if len(face.shape) == 3 else face)
        g48  = cv2.resize(gray, (48, 48)).astype(np.float32) / 255.0

        return self._compute_aus(g48)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _crop_face(img_bgr, bbox):
        if bbox is None:
            return img_bgr
        x, y, w, h = bbox
        # Inflate bounding box to include forehead / chin (matches CK+ framing)
        inf_w = int(w * 0.2)
        inf_h = int(h * 0.4)
        x = max(0, x - inf_w)
        y = max(0, y - inf_h)
        w = min(img_bgr.shape[1] - x, w + inf_w * 2)
        h = min(img_bgr.shape[0] - y, h + inf_h * 2)
        return img_bgr[y:y + h, x:x + w]

    @staticmethod
    def _grad(region):
        """Return (mean_grad_mag, vertical_ratio, horizontal_ratio, mean_intensity)."""
        if region.size == 0:
            return 0.0, 0.5, 0.5, 0.0
        gy = np.gradient(region, axis=0)
        gx = np.gradient(region, axis=1)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        mean_mag  = float(mag.mean())
        vert_r    = float(abs(gy).mean() / (mean_mag + 1e-6))
        horiz_r   = float(abs(gx).mean() / (mean_mag + 1e-6))
        mean_int  = float(region.mean())
        return mean_mag, vert_r, horiz_r, mean_int

    def _compute_aus(self, g: np.ndarray) -> dict:
        """
        g: (48, 48) float32 normalised face image.

        Anatomical row/col bands on a 48×48 face:
          rows  0-10  → forehead / brow
          rows  8-20  → eyes
          rows 18-28  → nose bridge + cheeks
          rows 26-36  → upper lip / mouth
          rows 34-48  → lower mouth / chin / jaw

          cols  0-20  → left side
          cols 14-34  → centre
          cols 28-48  → right side
        """
        # ── Region crops ─────────────────────────────────────────────────
        brow_l   = g[0:12,  0:24]   # left brow
        brow_r   = g[0:12, 24:48]   # right brow
        brow_c   = g[0:10, 14:34]   # inner / centre brow
        eye_l    = g[10:22,  2:22]  # left eye + lid
        eye_r    = g[10:22, 26:46]  # right eye + lid
        nose     = g[18:30, 14:34]  # nose bridge + tip
        cheek_l  = g[18:34,  0:18]  # left cheek
        cheek_r  = g[18:34, 30:48]  # right cheek
        ulip     = g[28:36, 10:38]  # upper lip
        llip     = g[34:44, 10:38]  # lower lip / mouth open
        jaw      = g[38:48,  6:42]  # jaw / chin

        # ── Gradient descriptors ──────────────────────────────────────────
        mg_bl, vr_bl, hr_bl, mi_bl = self._grad(brow_l)
        mg_br, vr_br, hr_br, mi_br = self._grad(brow_r)
        mg_bc, vr_bc, hr_bc, mi_bc = self._grad(brow_c)
        mg_el, vr_el, hr_el, mi_el = self._grad(eye_l)
        mg_er, vr_er, hr_er, mi_er = self._grad(eye_r)
        mg_ns, vr_ns, hr_ns, mi_ns = self._grad(nose)
        mg_cl, vr_cl, hr_cl, mi_cl = self._grad(cheek_l)
        mg_cr, vr_cr, hr_cr, mi_cr = self._grad(cheek_r)
        mg_ul, vr_ul, hr_ul, mi_ul = self._grad(ulip)
        mg_ll, vr_ll, hr_ll, mi_ll = self._grad(llip)
        mg_jw, vr_jw, hr_jw, mi_jw = self._grad(jaw)

        def c(v, lo=0.0, hi=1.0):
            return float(np.clip(v, lo, hi))

        # Scale factors determined empirically on CK+48 48×48 images
        # (gradient magnitudes typically 0.01-0.15 on normalised images)
        S = 6.0   # global gradient scale

        au = {}

        # AU01 — Inner Brow Raiser: upward gradient in centre brow
        au["AU01"] = c(vr_bc * mg_bc * S)

        # AU02 — Outer Brow Raiser: upward gradient in outer brows
        au["AU02"] = c(((vr_bl * mg_bl) + (vr_br * mg_br)) * 0.5 * S)

        # AU04 — Brow Lowerer: downward + inward, low brow position
        au["AU04"] = c((1.0 - mi_bc) * mg_bc * S)

        # AU05 — Upper Lid Raiser: strong gradient at eyelid boundary
        au["AU05"] = c(((mg_el + mg_er) * 0.5) * S)

        # AU06 — Cheek Raiser: elevated cheek intensity + gradient
        au["AU06"] = c(((mi_cl + mi_cr) * 0.5) * 0.8 + ((mg_cl + mg_cr) * 0.5) * 1.5)

        # AU07 — Lid Tightener: high horizontal gradient in eye region
        au["AU07"] = c(((hr_el * mg_el) + (hr_er * mg_er)) * 0.5 * S)

        # AU09 — Nose Wrinkler: strong gradient at nose root
        au["AU09"] = c(mg_ns * S * 0.8)

        # AU10 — Upper Lip Raiser: strong vertical gradient in upper lip
        au["AU10"] = c(vr_ul * mg_ul * S)

        # AU12 — Lip Corner Puller (Smile): horizontal gradient in upper lip
        #         combined with elevated cheek
        au["AU12"] = c((hr_ul * mg_ul * S * 0.6) + ((mi_cl + mi_cr) * 0.5 * 0.4))

        # AU14 — Dimpler: asymmetry in left vs right cheek gradient
        au["AU14"] = c(abs(mg_cl - mg_cr) * S)

        # AU15 — Lip Corner Depressor: downward gradient in lip corners
        #         (vertical ratio high, dark lower-mouth)
        au["AU15"] = c(vr_ll * mg_ll * S * (1.0 - mi_ul))

        # AU17 — Chin Raiser: gradient activity at jaw/chin area
        au["AU17"] = c(mg_jw * S * 0.7)

        # AU20 — Lip Stretcher: horizontal gradient in lower lip
        au["AU20"] = c(hr_ll * mg_ll * S)

        # AU23 — Lip Tightener: high gradient magnitude in upper lip with
        #         low opening (low lower-lip brightness means closed)
        au["AU23"] = c(mg_ul * S * (1.0 - mi_ll))

        # AU25 — Lips Part: brightness of lower mouth region (open=bright)
        au["AU25"] = c(mi_ll * 1.5)

        # AU26 — Jaw Drop: total gradient in lower-mouth + jaw area
        au["AU26"] = c((mg_ll + mg_jw) * 0.5 * S * 0.9)

        # AU28 — Lip Suck: low brightness in central mouth (lips pulled in)
        mouth_centre = g[30:40, 16:32]
        au["AU28"] = c(1.0 - float(mouth_centre.mean()) * 2.0)

        # AU45 — Blink: low gradient in eye region (closed lid = uniform)
        au["AU45"] = c(1.0 - ((mg_el + mg_er) * 0.5) * S)

        return au
