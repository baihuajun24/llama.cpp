cmake_minimum_required(VERSION 3.16)  # for CMAKE_CUDA_ARCHITECTURES # updated on 0527

# find_package(CUDAToolkit)

# if (CUDAToolkit_FOUND)
#     message(STATUS "CUDA Toolkit found")

enable_language(CUDA)
message(STATUS "Enabled CUDA language")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        elseif(GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "50;61;70;75;80")
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    enable_language(CUDA)

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        file(GLOB   SRCS "template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS "template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
    endif()

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_CUDA_GRAPHS)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
        add_compile_definitions(GGML_CUDA_F16)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()
    
    # Set default CUDA library directory
    if(NOT DEFINED CUDA_HOME)
        set(CUDA_HOME "/usr/local/cuda")
    endif()

    # Include headers and libraries
    include_directories(${CUDA_HOME}/include)
    link_directories(${CUDA_HOME}/lib64)

    # Link CUDA libraries
    if(GGML_STATIC)
        # Statically link libraries where available (not all platforms support static linking of all CUDA libs)
        if(WIN32)
            message(WARNING "Static linking of cublas is not supported on Windows CUDA 12.3.1+; using dynamic linking fallback.")
            target_link_libraries(ggml-cuda PRIVATE cudart cublas cublasLt)
        else()
            target_link_libraries(ggml-cuda PRIVATE cudart_static cublas_static cublasLt_static)
        endif()
    else()
        # Dynamic linking (default)
        target_link_libraries(ggml-cuda PRIVATE cudart cublas cublasLt)
    endif()

    # Optionally link against the CUDA driver API (libcuda.so), unless disabled
    if(NOT GGML_CUDA_NO_VMM)
        target_link_libraries(ggml-cuda PRIVATE cuda)
    endif()


    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math)

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
# else()
#     message(FATAL_ERROR "CUDA Toolkit not found")
# endif()
