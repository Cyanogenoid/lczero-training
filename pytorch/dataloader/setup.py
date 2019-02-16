import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension


setuptools.setup(
    name='lc0worker',
    ext_modules=[CppExtension(
        name='lc0worker',
        sources=[
            'worker.cpp',
            'proto/net.pb.cc',
            'proto/chunk.pb.cc',
            'lc0/src/chess/bitboard.cc',
            'lc0/src/chess/board.cc',
            'lc0/src/chess/position.cc',
            'lc0/src/neural/encoder.cc',
            'lc0/src/utils/logging.cc',
        ],
        include_dirs=[
            'lc0/src',
            '.'
        ],
        libraries=['protobuf'],
        #library_dirs=['/usr/lib', '/home/yan/anaconda3/envs/protoc/lib'],
        extra_compile_args=['-std=c++14', '-DNO_PEXT'],
#        extra_link_args=['-L/usr/lib'],
        #extra_link_args=['-L/home/yan/anaconda3/envs/protoc/lib'],
    )],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
