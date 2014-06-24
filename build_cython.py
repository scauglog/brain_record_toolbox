from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("kohonen_neuron_c", ["kohonen_neuron_c.pyx"]),
            Extension("brain_state_calculate_c", ["brain_state_calculate_c.pyx"]),
            Extension("cpp_file_tools_c", ["cpp_file_tools_c.pyx"])
            ]

setup(
  name = 'koho_neuron',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = ext_modules
)