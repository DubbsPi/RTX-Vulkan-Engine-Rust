glslc --target-spv=spv1.4 -fshader-stage=rgen shaders/raygen.rgen -o shaders/raygen.spv
glslc --target-spv=spv1.4 -fshader-stage=rmiss shaders/miss.rmiss -o shaders/miss.spv
glslc --target-spv=spv1.4 -fshader-stage=rchit shaders/closesthit.rchit -o shaders/closesthit.spv