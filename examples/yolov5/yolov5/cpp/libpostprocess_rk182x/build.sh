#!/bin/bash

RKNPU3_INCLUDE="../../../../3rdparty/rknpu3/include/"

export PATH=$PATH:/data0/toolchain/gcc/linux-x86/riscv64/Xuantie-900-gcc-elf-newlib-x86_64-V3.0.1/bin/

CC="riscv64-unknown-elf-gcc"
STRIP="riscv64-unknown-elf-strip"

CC_PATH=`which ${CC}`


if [ x"${CC_PATH}" == x""  ]
then
    echo "Can not find riscv64-unknown-elf-gcc, please set PATH"
fi

if [ $# -eq 1 ]; then
    if [ $1 == "yolov5" ]; then
        CFLAGS="${CFLAGS} -DENABLE_YOLOV5_POSTPROCESS"
    fi
    if [ $1 == "yolov8" ] || [ $1 == "yolov6" ]; then
        CFLAGS="${CFLAGS} -DENABLE_YOLOV8_POSTPROCESS"
    fi
else
    echo "Usage: $0 [yolov5|yolov6|yolov8]"
    exit 1
fi

CFLAGS="${CFLAGS}  -mcpu=c908v -mrvv-v0p10-compatible -march=rv64gcv -mabi=lp64d -O2 -g2 -mcmodel=medany -fpic -fno-plt  -D_POSIX_C_SOURCE=1 -I${RKNPU3_INCLUDE}"
LD_FLAGS="${LD_FLAGS} -mcpu=c908v -O2 -g2 -Wl,-r,-z,max-page-size=1024 -fpic -nostartfiles -nostdlib -static-libgcc -e 0"

TARGET_LIB="libpostprocess_$1_rk182x.so"

rm *.o *.so

${CC} -o yolov5_postprocess.o -c yolov5_postprocess.c ${CFLAGS} -I.
${CC} -o yolov8_postprocess.o -c yolov8_postprocess.c ${CFLAGS} -I.
${CC} -o rknn3_custom_op.o -c rknn3_custom_op.c ${CFLAGS} -I.

${CC} -o "${TARGET_LIB}"  ${LD_FLAGS} *.o 

${STRIP} --strip-unneeded -R .hash "${TARGET_LIB}"

echo "Build ${TARGET_LIB} done"

cp ${TARGET_LIB} prebuilt

echo "Copy ${TARGET_LIB} to prebuilt done"