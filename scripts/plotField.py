#!/bin/python3

import argparse
import logging
from pathlib import Path

import plot_utils.config as conf
import plot_utils.data_loader as loader
import plot_utils.utils as utils

from semtex_fieldplot.plot_field_data import plot_from_config


def process_config(config, config_path=None, cli_args=None):
    """
    Processes a single configuration (from YAML or CLI) and generates plots.
    """
    if not utils.validate_and_log_config(config, config_path):
        return

    if config.get("logging") == "INFO":
        logging.basicConfig(level=logging.INFO)
    elif config.get("logging") == "DEBUG":
        # logging.getLogger("plot_utils").setLevel(logging.DEBUG)
        # logging.getLogger("semtex_fieldio").setLevel(logging.DEBUG)
        # logging.getLogger("semtex_fieldplot").setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.WARNING)
    save_path = utils.determine_save_path(config, cli_args)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    file_names = utils.get_file_list(config.get("data_files", []))

    conf.set_font_sizes(config)
    conf.set_plot_style(config)

    try:
        plot_from_config(file_names, config, save_path)
    except Exception as e:
        print(f"Error plotting from config: {e}")

    utils.save_execution_command(save_path)
    utils.copy_config_to_save_path(config_path, save_path)


def main():
    """
    Main function to handle command-line arguments, load data, configure
    plotting, and generate plots.
    """
    parser = argparse.ArgumentParser(
        description="Plot data with optional YAML configuration."
    )

    # Allow multiple YAML config files
    parser.add_argument(
        "--config", type=str, nargs="+", help="Path to YAML configuration file(s)"
    )
    parser.add_argument(
        "--default-config",
        type=str,
        help="Path to the default YAML configuration",
        default="default_style.yaml",
    )

    args = parser.parse_args()

    # Load the default configuration if available
    default_config = {}
    if Path(args.default_config).exists():
        try:
            default_config = loader.load_yaml_config(args.default_config)
            print(f"✅ Loaded default configuration from {args.default_config}")
        except Exception as e:
            print(f"⚠️ Error loading default config {args.default_config}: {e}")

    # Load YAML config if specified
    if args.config:
        # Process multiple YAML config files
        config_list = utils.get_file_list(args.config)
        for config_path in config_list:
            try:
                specific_configs = loader.load_yaml_config(config_path)
            except Exception as e:
                print(f"⚠️ Error loading {config_path}: {e}")
            try:
                # If no default provided, look for "default" block in config
                if not default_config and "default" in specific_configs:
                    default_config = specific_configs["default"]
                    print("Use default config from YAML")

                # Process all the figure configs of one config file
                for config in specific_configs.get("figures", []):
                    # Merge each individual figure config with defaults
                    merged_config = utils.merge_configs(default_config, config)
                    process_config(merged_config, config_path, cli_args=args)
            except Exception as e:
                print(f"⚠️ Error processing {config_path}: {e}")
    else:
        # Process CLI inputs directly if no YAML is provided
        config = {}
        process_config(config, config_path=None, cli_args=args)


if __name__ == "__main__":
    main()
