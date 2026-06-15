set(TARGET_LOOKUP llama-lookup)
add_executable(${TARGET_LOOKUP} lookup.cpp)
install(TARGETS ${TARGET_LOOKUP} RUNTIME)
target_link_libraries(${TARGET_LOOKUP} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP} PRIVATE cxx_std_17)

set(TARGET_NGPLUS_VERIFIER llama-ngplus-verifier)
add_executable(${TARGET_NGPLUS_VERIFIER} ngplus-verifier.cpp)
install(TARGETS ${TARGET_NGPLUS_VERIFIER} RUNTIME)
target_compile_features(${TARGET_NGPLUS_VERIFIER} PRIVATE cxx_std_17)

set(NGPLUS_GEMMA4_LLAMA_ROOT "" CACHE PATH "Optional Gemma4-capable llama.cpp checkout for llama-ngplus-verifier")
if(NOT NGPLUS_GEMMA4_LLAMA_ROOT AND EXISTS "/Users/baihuajun/Documents/llama.cpp-upstream/build/bin/libllama.dylib")
    set(NGPLUS_GEMMA4_LLAMA_ROOT "/Users/baihuajun/Documents/llama.cpp-upstream")
endif()

if(NGPLUS_GEMMA4_LLAMA_ROOT)
    set(NGPLUS_GEMMA4_BIN_DIR "${NGPLUS_GEMMA4_LLAMA_ROOT}/build/bin")
    target_compile_definitions(${TARGET_NGPLUS_VERIFIER} PRIVATE NGPLUS_USE_UPSTREAM_GEMMA4)
    target_include_directories(${TARGET_NGPLUS_VERIFIER} BEFORE PRIVATE
        "${NGPLUS_GEMMA4_LLAMA_ROOT}/common"
        "${NGPLUS_GEMMA4_LLAMA_ROOT}/include"
        "${NGPLUS_GEMMA4_LLAMA_ROOT}/ggml/include"
        "${NGPLUS_GEMMA4_LLAMA_ROOT}/src")
    target_link_libraries(${TARGET_NGPLUS_VERIFIER} PRIVATE
        "${NGPLUS_GEMMA4_BIN_DIR}/libllama-common.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libllama.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libggml.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libggml-cpu.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libggml-blas.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libggml-metal.dylib"
        "${NGPLUS_GEMMA4_BIN_DIR}/libggml-base.dylib"
        ${CMAKE_THREAD_LIBS_INIT})
    if(APPLE)
        target_link_libraries(${TARGET_NGPLUS_VERIFIER} PRIVATE
            "-framework Foundation"
            "-framework Metal"
            "-framework MetalKit"
            "-framework Accelerate")
    endif()
    set_target_properties(${TARGET_NGPLUS_VERIFIER} PROPERTIES
        BUILD_RPATH "${NGPLUS_GEMMA4_BIN_DIR}"
        INSTALL_RPATH "${NGPLUS_GEMMA4_BIN_DIR}")
else()
    target_link_libraries(${TARGET_NGPLUS_VERIFIER} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
endif()

set(TARGET_LOOKUP_CREATE llama-lookup-create)
add_executable(${TARGET_LOOKUP_CREATE} lookup-create.cpp)
install(TARGETS ${TARGET_LOOKUP_CREATE} RUNTIME)
target_link_libraries(${TARGET_LOOKUP_CREATE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP_CREATE} PRIVATE cxx_std_17)

set(TARGET_LOOKUP_MERGE llama-lookup-merge)
add_executable(${TARGET_LOOKUP_MERGE} lookup-merge.cpp)
install(TARGETS ${TARGET_LOOKUP_MERGE} RUNTIME)
target_link_libraries(${TARGET_LOOKUP_MERGE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP_MERGE} PRIVATE cxx_std_17)

set(TARGET_LOOKUP_STATS llama-lookup-stats)
add_executable(${TARGET_LOOKUP_STATS} lookup-stats.cpp)
install(TARGETS ${TARGET_LOOKUP_STATS} RUNTIME)
target_link_libraries(${TARGET_LOOKUP_STATS} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP_STATS} PRIVATE cxx_std_17)

set(TARGET_NGRAM lookup-ngram)
add_executable(${TARGET_NGRAM} lookup-ngram.cpp)
install(TARGETS ${TARGET_NGRAM} RUNTIME)
target_link_libraries(${TARGET_NGRAM} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NGRAM} PRIVATE cxx_std_17)

# Add the new ngram-index executables
set(TARGET_NGRAM_INDEX llama-lookup-ngram-index)
add_executable(${TARGET_NGRAM_INDEX} lookup-ngram-index.cpp)
install(TARGETS ${TARGET_NGRAM_INDEX} RUNTIME)
target_link_libraries(${TARGET_NGRAM_INDEX} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NGRAM_INDEX} PRIVATE cxx_std_17)

# Add the new ngram-table executable
set(TARGET_NGRAM_TABLE llama-lookup-ngram-table)
add_executable(${TARGET_NGRAM_TABLE} lookup-ngram-table.cpp)
install(TARGETS ${TARGET_NGRAM_TABLE} RUNTIME)
target_link_libraries(${TARGET_NGRAM_TABLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NGRAM_TABLE} PRIVATE cxx_std_17)

# Add the new pld-search executable
set(TARGET_NGRAM_TABLE llama-lookup-pld-search)
add_executable(${TARGET_NGRAM_TABLE} pld-search.cpp)
install(TARGETS ${TARGET_NGRAM_TABLE} RUNTIME)
target_link_libraries(${TARGET_NGRAM_TABLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NGRAM_TABLE} PRIVATE cxx_std_17)

# (0707 added) Add the new build-suffix-array executable when the source is present.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/build-suffix-array.cpp")
    set(TARGET_BUILD_SUFFIX_ARRAY llama-build-suffix-array)
    add_executable(${TARGET_BUILD_SUFFIX_ARRAY} build-suffix-array.cpp ../../common/ngram-suffix.cpp)
    install(TARGETS ${TARGET_BUILD_SUFFIX_ARRAY} RUNTIME)
    target_link_libraries(${TARGET_BUILD_SUFFIX_ARRAY} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET_BUILD_SUFFIX_ARRAY} PRIVATE cxx_std_17)
endif()

# (0708 added) Add the new sample-ngram-table executable
set(TARGET_SAMPLE_NGRAM_TABLE llama-sample-ngram-table)
add_executable(${TARGET_SAMPLE_NGRAM_TABLE} sample-ngram-table.cpp)
install(TARGETS ${TARGET_SAMPLE_NGRAM_TABLE} RUNTIME)
target_link_libraries(${TARGET_SAMPLE_NGRAM_TABLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_SAMPLE_NGRAM_TABLE} PRIVATE cxx_std_17)

# Add the new sample-ngram-multiple executable  
set(TARGET_SAMPLE_NGRAM_MULTIPLE llama-sample-ngram-multiple)
add_executable(${TARGET_SAMPLE_NGRAM_MULTIPLE} sample-ngram-multiple.cpp)
install(TARGETS ${TARGET_SAMPLE_NGRAM_MULTIPLE} RUNTIME)
target_link_libraries(${TARGET_SAMPLE_NGRAM_MULTIPLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_SAMPLE_NGRAM_MULTIPLE} PRIVATE cxx_std_17)

# (0710 added) Add the new lookup-suffix executable
set(TARGET_LOOKUP_SUFFIX llama-lookup-suffix)
add_executable(${TARGET_LOOKUP_SUFFIX} lookup-suffix.cpp ../../common/ngram-suffix.cpp)
install(TARGETS ${TARGET_LOOKUP_SUFFIX} RUNTIME)
target_link_libraries(${TARGET_LOOKUP_SUFFIX} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP_SUFFIX} PRIVATE cxx_std_17)

# --- Skip building build-ngram-index (temporary) ---
# set(TARGET_BUILD_NGRAM_INDEX build-ngram-index)
# add_executable(${TARGET_BUILD_NGRAM_INDEX} build-ngram-index.cpp)
# install(TARGETS ${TARGET_BUILD_NGRAM_INDEX} RUNTIME)
# target_link_libraries(${TARGET_BUILD_NGRAM_INDEX} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET_BUILD_NGRAM_INDEX} PRIVATE cxx_std_17)

# --- Skip building llama-lookup-rest (temporary) ---
# set(TARGET_REST llama-lookup-rest)
# add_executable(${TARGET_REST} lookup-rest.cpp)
# install(TARGETS ${TARGET_REST} RUNTIME)

# set(DRAFTRETRIEVER_DIR "${CMAKE_BINARY_DIR}/bin")
# set(DRAFTRETRIEVER_CXX_LIB "${DRAFTRETRIEVER_DIR}/cxxbridge1.lib")
# set(DRAFTRETRIEVER_LIB "${DRAFTRETRIEVER_DIR}/draftretriever.dll.lib")

# add_library(draftretriever_cxx STATIC IMPORTED)
# set_target_properties(draftretriever_cxx PROPERTIES 
#     IMPORTED_LOCATION "${DRAFTRETRIEVER_CXX_LIB}"
# )

# add_library(draftretriever_lib SHARED IMPORTED)
# set_target_properties(draftretriever_lib PROPERTIES 
#     IMPORTED_IMPLIB "${DRAFTRETRIEVER_LIB}"
#     IMPORTED_LOCATION "${DRAFTRETRIEVER_DIR}/draftretriever.dll"
# )

# target_link_libraries(${TARGET_REST} PRIVATE 
#     common 
#     llama 
#     rest_index
#     draftretriever_cxx
#     draftretriever_lib
# )
# target_compile_features(${TARGET_REST} PRIVATE cxx_std_17)
