Please modify the existing Python code of the 'Poraque' project to implement the following performance enhancements, features, and optimizations. Ensure that the code remains modular, well-documented, and adheres to clean coding practices. Use python 3.14 in conda environment jit (conda activate jit).

Here are the specific requirements for the modification:

1. Parallel Computing (MPI / OpenMP):
   - Integrate parallel computing capabilities to speed up heavy calculations.
   - You may use OpenMP (via Cython/C++ extensions) or MPI (via mpi4py), whichever is more appropriate for the current architecture of the codebase. Optimize the most computationally intensive loops or tasks.

2. Libxc Integration (XC Functionals):
   - Integrate the 'libxc' library to handle Exchange-Correlation (XC) functionals.
   - Specifically implement support for the PBE and PBEsol functionals using libxc bindings.

3. Numba & JIT Testing (Python >= 3.14):
   - Add performance and regression tests utilizing Numba's Just-In-Time (JIT) compilation.
   - Ensure compatibility with Python 3.14+ JIT features. Create a test suite that validates the correctness and speedup of these JIT-compiled functions.

4. Execution Timing & Profiling:
   - Implement a timing/profiling module that measures and logs the execution time of each individual module/component.
   - The timing summary should be printed or logged at the end of the execution.

5. Dynamic Output Argument:
   - Modify the main 'Poraque' class or function signature to accept an 'output' argument for specifying the path of the standard output file.
   - It should default to "output.txt" (e.g., `output="output.txt"`), and all standard execution logs/results should be redirected to this file.

Please provide the updated code sections, explanations of the changes made, and instructions on how to run the new test suite.