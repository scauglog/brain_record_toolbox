from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("kohonen_neuron_c", ["kohonen_neuron_c.pyx"])]

setup(
  name = 'koho_neuron',
  cmdclass = {'build_ext': build_ext},
  include_dirs = [np.get_include()],         # <---- New line
  ext_modules = ext_modules
)