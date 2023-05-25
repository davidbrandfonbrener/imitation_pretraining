"""Example of generating data for point mass."""
import fire
import point_mass_config
from imitation_pretraining.experiments.data_generation import point_mass


def main():
    # Generate pretraining data
    config = point_mass_config.get_config()
    config["pretrain"] = True
    point_mass.PointMassGenerator(config).generate()

    # Generate field data
    config["pretrain"] = False
    config["episodes_per_seed"] = config["episodes"]  # Only 1 context for finetuning
    point_mass.PointMassGenerator(config).generate()


if __name__ == "__main__":
    fire.Fire(main)
