import os
cimport openmp


cdef _get_effective_num_threads(n_threads=None):
    """
    Based on Scikit-learn:  scikit-learn/sklearn/utils/_openmp_helpers.pyx
    Determine the effective number of threads to be used for OpenMP calls
      - For ``n_threads = None``,
        - if the ``OMP_NUM_THREADS`` environment variable is set, return
          ``openmp.omp_get_max_threads()``
        - otherwise, return the minimum between ``openmp.omp_get_max_threads()``
          and the number of cpus, taking cgroups quotas into account. Cgroups
          quotas can typically be set by tools such as Docker.
        The result of ``omp_get_max_threads`` can be influenced by environment
        variable ``OMP_NUM_THREADS`` or at runtime by ``omp_set_num_threads``.
      - For ``n_threads > 0``, return this as the maximal number of threads for
        parallel OpenMP calls.
      - For ``n_threads < 0``, return the maximal number of threads minus
        ``|n_threads + 1|``. In particular ``n_threads = -1`` will use as many
        threads as there are available cores on the machine.
      - Raise a ValueError for ``n_threads = 0``.
      If scikit-learn is built without OpenMP support, always return 1.
    """
    if n_threads == 0:
        raise ValueError("n_threads = 0 is invalid")

    TOFU_OPENMP_ENABLED = check_if_openmp_installed()
    if TOFU_OPENMP_ENABLED:
        if os.getenv("OMP_NUM_THREADS"):
            # Fall back to user provided number of threads making it possible
            # to exceed the number of cpus.
            max_n_threads = openmp.omp_get_max_threads()
        else:
            max_n_threads = min(openmp.omp_get_max_threads(),
                                len(os.sched_getaffinity(0)))

        if n_threads is None:
            return max_n_threads
        elif n_threads < 0:
            return max(1, max_n_threads + n_threads + 1)

        return min(n_threads, max_n_threads)
    else:
        # OpenMP disabled at build-time => sequential mode
        return 1




def check_if_openmp_installed():
    # Check if openmp available
    # see http://openmp.org/wp/openmp-compilers/
    omp_test = r"""
    #include <omp.h>
    #include <stdio.h>
    int main() {
    #pragma omp parallel
    printf("Hello from thread %d, nthreads %d\n", omp_get_thread_num(),
           omp_get_num_threads());
    }
    """
    import tempfile
    import platform
    import shutil
    import subprocess

    if platform.system() == "Windows":
        return False

    tmpdir = tempfile.mkdtemp()
    curdir = os.getcwd()
    os.chdir(tmpdir)

    filename = r"test.c"
    with open(filename, "w") as file:
        file.write(omp_test)
    with open(os.devnull, "w") as fnull:
        result = subprocess.call(
            ["cc", "-fopenmp", filename], stdout=fnull, stderr=fnull
        )

    os.chdir(curdir)
    # clean up
    shutil.rmtree(tmpdir)

    return  not result
