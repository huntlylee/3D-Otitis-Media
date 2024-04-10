# 🖥️ A Three-Dimensional and Explainable AI Model for Automatic Evaluation of Chronic Otitis Media (COM) 👂

This repository contains the artificial intelligence model for automatic evaluation of Chronic Otitis Media (COM) using temporal bone CT scans. The model is designed to assist clinicians and researchers in diagnosing COM and its subtypes, such as chronic suppurative otitis media (CSOM) and cholesteatoma, in an explainable fashion.

## Description

The COM evaluation model employs a convolutional neural network (CNN) architecture optimized for analyzing three-dimensional (3D) medical imaging data. It provides an end-to-end diagnostic tool capable of processing raw CT images to classify and differentiate various forms of COM. The heatmap technique is employed to exhibit the rationale of the AI model in making predictions.

## Repository Primary Contents 

- `tutoral.ipynb`: Jupyter notebooks with examples and usage instructions.
- `run_full_workflow.py`: the python script that take DICOM image and generate output.
- `Source_code/`: Source code and trained models generated during the development phase, for validation purpose
- `scripts/`: Source code for model evaluation, and deployment.
- `Model_weights/`: the pretrained deep learning model weights for application.
- `environment.yml`: List of Python packages installed with conda.
- `requirements.txt`: List of Python dependencies for setting up the environment.
  
## Installation

To set up your environment to run the model, follow these steps:

1. Clone the repository:

`git clone https://github.com/huntlylee/3D-Otitis-Media.git`

2. Navigate to the repository directory:

`cd 3D-Otitis-Media`

3. Install the required Python packages:

`conda env create -f environment.yml`
`conda activate otitis`
`pip install -r requirements.txt`

## Tutorial

Please walk through the tutorial.ipynb to get detailed information on the workflow execution and how the arguments are set up and utilized.

## Usage

The temporal bone CT images should be in the [DICOM](https://en.wikipedia.org/wiki/DICOM) format. It is ideally placed in a subfolder within the repository folder. 

Run the full workflow script with the following command:

```bash
Usage: python run_full_workflow.py [OPTIONS]
Set the path configurations with specified command-line arguments.

Options:
    --out_root_folder=PATH     Path to the output root folder. Default is 'output'.
    --scan_path=PATH    Full path to the folder containing the individual's CT scan. For example: 'CT_images/p00726056-231124'.
    --target_side=SIDE        Target side. Options are 'Left' or 'Right'. Default is 'Left'.

Example:
    python run_full_workflow.py --out_root_folder=output --scan_path=CT_images/p00726056-231124 --target_side=Left
```
## Sample output
![Sample Output](output/p00726056-231124%20Left.png)

![Sample Output](output/P00085041-231029%20Left.png)

## Contributing

Contributions to improve the model or its implementation are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## Citation

If you use this model or the associated code in your research, please cite the following paper:

Chen B, Li Y, Sun Y, Sun H, Wang Y, Lyu J, Guo J, Cheng Y, Niu X, Yang L, Xu J, Yang J, Huang Y, Chi F, Liang B, Ren D. Three-Dimensional and Explainable Deep Learning for Evaluation of Chronic Otitis Media Based on Temporal Bone Computed Tomography: Development Study
JMIR Preprints. 09/08/2023:51706. DOI: 10.2196/preprints.51706. URL: https://preprints.jmir.org/preprint/51706

## License

This project is licensed under the [GPL-3.0 license](LICENSE.txt).

## Contact

For any queries or requests regarding the dataset or further collaboration, please contact the corresponding authors.

## Acknowledgments

We would like to thank all contributors to this project and the institutions that provided the datasets used for training the model.
