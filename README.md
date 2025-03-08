# bls_cuda
An implementation of Box-Least-Squares transit search using CUDA.

How does it work:

The idea is to send a few thousand jobs at a time to the GPU.  We loop through tossing blocks of frequencies to compute BLS at and collecting the results as we good.  I've tested with Kepler data scanning over 700~000 frequencies from 0.5 days to the length of the dataset without issues.  You can't just assign one frequency per thread or you will quickly run out of VRAM.  

## Getting Started
To get started with this project, follow these steps:
1. Clone the repository: `git clone https://github.com/jasonfrowe/bls_cuda.git`
2. Install required dependencies:  numba, numba-cuda, matplotlib (and others)
3. You can either have a look at the Jupyter Notebook (bls_cuda.ipynb) or try running transitfind5_gpu.py

## Commandline Usage
1. Make sure transitfind5_gpu.py is executable
2. `./transitfind5_gpu.py filename.txt`
3. filename.txt is a simple space delimited file with time and flux in two columns.
4. if everything works, then a plot should show up showing the results of your search
5. try: `./transitfind5_gpu.py` to see command usage

## Contributing
If you'd like to contribute to this project, go for it! There are a number of to-dos 
1. Code speed can likely be made much faster.  (shared memory vs global memory)
2. Better choices of blocks and threads-per-block needs to be explored
3. Making the code base into an installable package
4. and much more.. 

## License
This project is licensed under the GNU General Public License (GPL) version 3 or later.

## Acknowledgments
Thank you to Canada Research Chairs, NSERC Discovery, Digital Alliance Canada, Calcul Quebec, FRQNT for financial and hardware support.

This code was initially developed during the Bishop's University Winter Reading Week, making good use of profession development resources. 

If you find these codes useful please reference:  
Rowe et al. 2014 ApJ, 784, 45   
Rowe et al. 2015 ApJs, 217, 16  

## Change Log
2025/03/08 : Initial Update
