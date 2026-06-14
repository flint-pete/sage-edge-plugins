"""
BioCLIP2 Species Classifier Plugin for Sage/Waggle
Captures camera frames, classifies biological organisms using BioCLIP2
(imageomics/bioclip-2), and publishes taxonomy predictions.

BioCLIP2 is a CLIP model fine-tuned on the TreeOfLife-200M dataset
covering 450K+ species with significantly improved accuracy over v1.
It classifies at any taxonomic rank:
Kingdom > Phylum > Class > Order > Family > Genus > Species.

Default model: BioCLIP2 (hf-hub:imageomics/bioclip-2)
  - Architecture: SigLIP2-based vision transformer (~430M params)
  - Training data: TreeOfLife-200M (200M+ biological images)
  - ~2.5 GB GPU memory at inference
  - Fits easily in 128GB unified memory (DGX Spark / Sage Thor)

Measurement topics:
  env.species.<rank>           — top predicted taxon name at chosen rank
  env.species.<rank>.confidence — confidence score (0-1)
  env.species.top5             — JSON of top-5 predictions
  upload                       — original camera image
"""
import argparse
import json
import logging
import tempfile
import time

import cv2
from PIL import Image

from bioclip import Rank
from bioclip.predict import TreeOfLifeClassifier

from waggle.plugin import Plugin
from waggle.data.vision import Camera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bioclip-species")

# Map string rank names to bioclip.Rank enum
RANK_MAP = {
    "Kingdom": Rank.KINGDOM,
    "Phylum": Rank.PHYLUM,
    "Class": Rank.CLASS,
    "Order": Rank.ORDER,
    "Family": Rank.FAMILY,
    "Genus": Rank.GENUS,
    "Species": Rank.SPECIES,
}
RANK_NAMES = list(RANK_MAP.keys())


class BioCLIP2Classifier:
    """BioCLIP2 species classifier for Sage edge nodes.

    Uses the pybioclip library's TreeOfLifeClassifier which handles
    model loading, text embeddings, and taxonomic classification.
    """

    def __init__(self, rank: str = "Class",
                 model_str: str = "hf-hub:imageomics/bioclip-2"):
        if rank not in RANK_MAP:
            raise ValueError(f"Invalid rank '{rank}'. Must be one of: {RANK_NAMES}")
        self.rank = rank
        self.rank_enum = RANK_MAP[rank]
        self.model_str = model_str

        logger.info("Loading BioCLIP2 classifier (model=%s, rank=%s)...",
                     model_str, rank)
        self.classifier = TreeOfLifeClassifier(model_str=model_str)
        logger.info("BioCLIP2 classifier loaded successfully")

    def classify(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        """
        Classify an image at the configured taxonomic rank.
        Returns list of {name, confidence} dicts, sorted descending.

        pybioclip returns a list of dicts with keys like:
          file_name, kingdom, phylum, class, order, family, genus,
          species_epithet, species, common_name, score
        """
        results = self.classifier.predict(
            images=[image],
            rank=self.rank_enum,
            k=top_k,
        )

        # Build a display name from the rank-level key
        rank_key = self.rank.lower()
        if rank_key == "species":
            # Species rank has a 'species' key with binomial name
            predictions = [
                {"name": r.get("species", r.get("genus", "Unknown")),
                 "confidence": float(r["score"])}
                for r in results
            ]
        else:
            predictions = [
                {"name": r.get(rank_key, "Unknown"),
                 "confidence": float(r["score"])}
                for r in results
            ]

        return predictions[:top_k]


# ── main loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BioCLIP2 Species Classifier for Sage")
    parser.add_argument("--stream", default="bottom_camera",
                        help="Camera stream name or RTSP URL")
    parser.add_argument("--rank", default="Class",
                        choices=RANK_NAMES,
                        help="Taxonomic rank for classification")
    parser.add_argument("--model", default="hf-hub:imageomics/bioclip-2",
                        help="BioCLIP model string (default: BioCLIP2)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between captures")
    parser.add_argument("--min-confidence", type=float, default=0.1,
                        help="Minimum confidence to publish")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to publish")
    parser.add_argument("--continuous", default="Y",
                        help="Y = loop, N = single-shot")
    args = parser.parse_args()

    classifier = BioCLIP2Classifier(
        rank=args.rank,
        model_str=args.model,
    )
    camera = Camera(args.stream)

    with Plugin() as plugin:
        logger.info("Plugin started — stream=%s, rank=%s, model=%s, interval=%ds",
                     args.stream, args.rank, args.model, args.interval)
        while True:
            try:
                sample = camera.snapshot()
                frame = sample.data  # numpy BGR

                # Convert BGR -> RGB -> PIL
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                predictions = classifier.classify(pil_image, top_k=args.top_k)

                if predictions and predictions[0]["confidence"] >= args.min_confidence:
                    top = predictions[0]

                    # Publish top prediction
                    rank_lower = args.rank.lower()
                    plugin.publish(
                        f"env.species.{rank_lower}",
                        top["name"],
                        timestamp=sample.timestamp,
                        meta={"camera": args.stream, "rank": args.rank,
                              "model": args.model},
                    )
                    plugin.publish(
                        f"env.species.{rank_lower}.confidence",
                        top["confidence"],
                        timestamp=sample.timestamp,
                        meta={"camera": args.stream, "rank": args.rank},
                    )

                    # Publish top-5 as JSON
                    plugin.publish(
                        f"env.species.top5",
                        json.dumps(predictions),
                        timestamp=sample.timestamp,
                        meta={"camera": args.stream, "rank": args.rank},
                    )

                    logger.info("Top prediction: %s (%.4f)", top["name"], top["confidence"])
                    for i, p in enumerate(predictions[1:], 2):
                        logger.info("  #%d: %s (%.4f)", i, p["name"], p["confidence"])

                    # Upload source image
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    cv2.imwrite(tmp.name, frame)
                    plugin.upload_file(tmp.name, timestamp=sample.timestamp,
                                       meta={"camera": args.stream,
                                             "top_species": top["name"],
                                             "confidence": str(top["confidence"])})
                else:
                    logger.info("No confident prediction (top=%.4f, threshold=%.2f)",
                                predictions[0]["confidence"] if predictions else 0,
                                args.min_confidence)

            except Exception:
                logger.exception("Classification error")

            if args.continuous != "Y":
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
