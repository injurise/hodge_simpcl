# Hodge-Aware Simplicial Contrastive Learning 

## Introduction
This repository contains the code implementation for the method proposed in the paper "Hodge-Aware Contrastive Learning". 
There we introduce a simplicial contrastive learning method that allows to learn effective representations for edge flow data.
Here we provide an example implementation for the trajectory dataset discussed in the paper.

## Installation

Clone the repository and install the packages from the requirements text file into your virtual environment:
```bash
pip install -r requirements.txt
```

## Detailed Description & Usage

The paper mainly introduces two contributions (1) a spectrally-optimized dropout augmentation and (2) a spectrally-reweighted loss.
For (1) the dropout probabilities have to be optimized first before training the contrastive model (we implemented it as a two-step procedure).
The code for the optimization of the dropout probabilities can be found in the `src/preprocessing/calculate_spectral_probabilities/run_calc_prob.py` file.
This will compute the probabilities and store them in the `data` directory. We have already provided a file with example precalculated probabilities there. 

As soon as the probabilities are calculated we can train a simplicial contrastive model. The code for this can be found in the `src/experiments/edge_flow_class/efc_sscl.py` file.
We use the sacred library for logging and you can simply run the file with specified config parameters from the command line. For example:
```bash
python efc_sscl.py with dataset="trajectories" model_type="CSCN" lr=0.001 weight_decay=0.0001 
```

To implement the code for contrastive learning we followed a modular approach and use a repository structure inspired by PyGCL (https://github.com/PyGCL/PyGCL). The repository contains the following main directories:
- **data**: Contains the simulated trajectory dataset used in the paper. 
- **logs**: Contains the logs of the training process of the contrastive models and of the optimization of the probabilities. 
- **src/augmentors**: Contains classes that implement the augmentation methods used (Identity, Edge Drop, Spectral Edge Drop).
- **src/contrast_models**: Contains classes that implement the contrastive learning frameworks (e.g. a Dual-Branch Architecture)
- **src/experiments**: Contains the scripts to train & evaluate a contrastive model on the trajectory dataset. 
                       Furthermore, in the baselines folder it contains the code to train & evaluate a supervised simplicial network.
- **src/preprocessing**: Contain the code to simulate the trajectory dataset and the code to compute the spectrally-optimized dropout probabilities.

## Execution on Slurm Clusters

If you run the code on a slurm cluster we recommend using the submitit library (https://github.com/facebookincubator/submitit).
An example code snippet to submit multiple parallel runs from a list of config dictionaries (config_dicts) is shown below:

```python
    import submitit

    executor = submitit.AutoExecutor(folder=YOUR_OUTPUT_FOLDER_PATH)

    executor.update_parameters(slurm_time=60,
                               slurm_job_name="efc",
                               cpus_per_task=4,
                               slurm_mem_per_cpu="6G",
                               slurm_array_parallelism=10)
    jobs = []
    with executor.batch():
        for conf_dict in config_dicts:
            command_list = ["python",
                            "src/experiments/edge_flow_class/efc_sscl.py",
                            "-F",
                            YOUR_LOG_FOLDER_PATH,
                            "with"]
            for key, value in conf_dict.items():
                command_list.append(f"{key}={value}")
            function = submitit.helpers.CommandFunction(command_list)
            job=executor.submit(function)
            jobs.append(job)


```

## Contact 
For any inquiries, please send me an email. Maybe I can help :)

<a href="mailto:alexander.j.moellers@gmail.com?"><img src="https://img.shields.io/badge/gmail-%23DD0031.svg?&style=for-the-badge&logo=gmail&logoColor=white"/></a>

## License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Citation 
Please cite our paper:
```
@inproceedings{ICCASP2023hodgeaware,
      title={Hodge-Aware Contrastive Learning}, 
      author={Alexander Möllers, Alexander Immer and Vincent Fortuin and Elvin Isufi},
      year={2024},
      booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing }}
```

## Credits 

The code for the contrastive learning framework is inspired by the PyGCL repository (https://github.com/PyGCL/PyGCL).
Some of the code for the simplicial networks is adjusted from https://github.com/domenicocinque/spm and some utility functions 
to process Simplicial Complexes are taken from https://github.com/ggoh29/Simplicial-neural-network-benchmark. 
We additionally specified this in the code whenever applicable.