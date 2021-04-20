# cython: language_level=3

import os
from .openmp_enabled import is_openmp_enabled

IF TOFU_OPENMP_ENABLED:
    cimport openmp

cpdef get_effective_num_threads(n_threads=None):
    """
    Based on Scikit-learn:  scikit-learn/sklearn/utils/_openmp_helpers.pyx
    Determine the effective number of threads to be used for OpenMP calls
      - For ``n_threads = None``,
        - if the ``OMP_NUM_THREADS`` environment variable is set, return
          ``openmp.omp_get_max_threads()``
        - otherwise return the minimum between ``openmp.omp_get_max_threads()``
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

    local_openmp_enabled = is_openmp_enabled()
    assert local_openmp_enabled == TOFU_OPENMP_ENABLED

    IF TOFU_OPENMP_ENABLED:

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
    ELSE:
        # OpenMP disabled at build-time => sequential mode
        return 1
