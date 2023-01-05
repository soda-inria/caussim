import argparse
from caussim.experiences.utils import consolidate_xps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xp_name",
        type=str,
        default=None,
        help="Folder of raw individual xps to consolidate",
    )
    parser.add_argument(
        "--xp_save",
        type=str,
        default=None,
        help="Folder of consolidated xps to consolidate with new raw xps. By defautl it save to <xp_name>_save.",
    )
    config, _ = parser.parse_known_args()
    config = vars(config)

    consolidate_xps(config["xp_name"], config["xp_save"])
