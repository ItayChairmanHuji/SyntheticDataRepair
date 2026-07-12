# Synthetic Data Repair

This is the code base that implements the paper "Repairing Privately Generated Synthetic Data with Integrity and
Statistical Considerations". The code is written in Python and is based on the Hydra framework.

## Basic Setup

For this code to work, you need to setup an environment. You can do it with a simple virtual environment.
Run `pyhton -m venv .venv` and then `source .venv/bin/activate` on Linux/Mac or `. .venv/Scripts/activate` on Windows.

After that, you need to install the dependencies. Run `pip install -r requirements.txt`.

## Running the Code

The code is based on the Hydra framework, so you can run it with `python -m main`.
Parameters can be overridden using both the command line and by editing the config files themselves.
For example if you want to run the hybrid repair you can run
`PYTHONPATH=$(pwd) python -m main --config-path hybrid/configs --config-name config --override input_path="<your_path> output_path="<your_path>"`
Another option is to edit the config files directly/ create a config file and pass it to the command line.
For example, you can create a config file called `my_config.yaml` and pass it to the command line with
`--config-file my_config.yaml`.
It is recommended to run the code from the root directory of the project, and reference configuration file in order to
avoid issues with the relative paths and imports.

## Resources

The `resources.zip` file contains the private datasets and denial constraints used in the paper experiments and several
representative synthetic generation models and marginals. All can be generated using the scripts in the codebase. 

## Kamino
The current code base does not support generating synthetic data using the Kamino algorithm. 
The generation of such data was done using [the official code of Kamino](https://github.com/cgebest/kamino)