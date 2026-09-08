glslc -I./shaders/include --target-spv=spv1.4 -fshader-stage=rgen shaders/raygen.rgen -o shaders/raygen.spv
glslc -I./shaders/include --target-spv=spv1.4 -fshader-stage=rmiss shaders/miss.rmiss -o shaders/miss.spv
glslc -I./shaders/include --target-spv=spv1.4 -fshader-stage=rchit shaders/closesthit.rchit -o shaders/closesthit.spv

glslc -I./shaders/include --target-spv=spv1.4 -fshader-stage=rmiss shaders/shadow.rmiss -o shaders/shadow.spv

glslc -I./shaders/include --target-spv=spv1.4 shaders/skin.comp -o shaders/skin.spv

glslc -I./shaders/include --target-spv=spv1.4 -fshader-stage=rahit shaders/cutout.rahit -o shaders/cutout.spv
