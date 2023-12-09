# Setup
1. Create a conda environment (optional)
2. Install the requirements with `pip install -r requirements.txt`
3. Clone the data repository with `git clone https://github.com/muxspace/facial_expressions.git`
4. For GPU compute, install [CUDA](https://developer.nvidia.com/cuda-downloads) and PyTorch with CUDA support ([instructions](https://pytorch.org/get-started/locally/) for instructions)

# Usage

## Testing
Run `python -m unittest discover -s testing -p 'test_*.py'`. This will run all the tests in the `testing` directory that start with `test_`.