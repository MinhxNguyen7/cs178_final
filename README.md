# Setup

## Local
1. Create a conda environment (optional)
2. Install the requirements with `pip install -r requirements.txt`
3. Clone this repository with `git clone https://github.com/minhxNguyen7/cs178_final.git`
4. Clone the data repository with `git clone https://github.com/muxspace/facial_expressions.git`
5. For GPU compute, install [CUDA](https://developer.nvidia.com/cuda-downloads) and PyTorch with CUDA support ([instructions](https://pytorch.org/get-started/locally/) for instructions)

## Colab
Note: I haven't tested this extensively, so let me know if there are any issues.
1. Install the requirements with `pip install -r requirements.txt`
2. Clone the data repository with `git clone https://github.com/muxspace/facial_expressions.git`
3. Clone this repository with `git clone https://github.com/minhxNguyen7/cs178_final.git`
   1. Move all files from this repository (`cs178_final`) to the root directory with `mv cs178_final/* .` from the root directory.

# Usage

## Testing
Run `python -m unittest discover -s testing -p 'test_*.py'`. This will run all the tests in the `testing` directory that start with `test_`.