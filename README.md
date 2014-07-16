brain_record_toolbox
====================

python and matlab script to process brain signal

script made for python 2.7 32bit and cython 0.20

require numpy pyYAML six dateutil pyparsing pytz matplotlib scipy

if you want to use X_c you must compile it :
>python build_cython.py build_ext --inplace

otherwise replace X_c by X
