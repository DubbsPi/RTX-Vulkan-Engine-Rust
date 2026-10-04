#!/bin/bash

G_FLAG=""

while getopts "g" opt; do
    case $opt in
        g)
            G_FLAG="-g"
            ;;
    esac
done


glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 -fshader-stage=rgen shaders/raygen.rgen -o shaders/raygen.spv
glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 -fshader-stage=rmiss shaders/miss.rmiss -o shaders/miss.spv
glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 -fshader-stage=rchit shaders/closesthit.rchit -o shaders/closesthit.spv

glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 -fshader-stage=rmiss shaders/shadow.rmiss -o shaders/shadow.spv
glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 -fshader-stage=rahit shaders/cutout.rahit -o shaders/cutout.spv

glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 shaders/skin.comp -o shaders/skin.spv

glslc $G_FLAG -O -I./shaders/include --target-env=vulkan1.4 --target-spv=spv1.6 shaders/denoise.comp -o shaders/denoise.spv
