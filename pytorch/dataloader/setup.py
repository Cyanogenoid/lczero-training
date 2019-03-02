import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension


setuptools.setup(
    name='lczero_training_worker',
    ext_modules=[CppExtension(
        name='lczero_training_worker',
        sources=[
            'worker.cpp',
        ],
        include_dirs=[
            '../../libs/flatbuffers/include'
        ],
    )],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
