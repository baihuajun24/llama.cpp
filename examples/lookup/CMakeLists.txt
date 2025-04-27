set(TARGET_LOOKUP llama-lookup)
add_executable(${TARGET_LOOKUP} lookup.cpp)
install(TARGETS ${TARGET_LOOKUP} RUNTIME)
target_link_libraries(${TARGET_LOOKUP} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_LOOKUP} PRIVATE cxx_std_17)

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

# Add the lookup-rest executable
set(TARGET_REST llama-lookup-rest)
add_executable(${TARGET_REST} lookup-rest.cpp)
install(TARGETS ${TARGET_REST} RUNTIME)

# Define DraftRetriever library paths - point to the bin directory where you copied the files
set(DRAFTRETRIEVER_DIR "${CMAKE_BINARY_DIR}/bin")
set(DRAFTRETRIEVER_CXX_LIB "${DRAFTRETRIEVER_DIR}/cxxbridge1.lib")
set(DRAFTRETRIEVER_LIB "${DRAFTRETRIEVER_DIR}/draftretriever.dll.lib")

# Add them as imported libraries with absolute paths
add_library(draftretriever_cxx STATIC IMPORTED)
set_target_properties(draftretriever_cxx PROPERTIES 
    IMPORTED_LOCATION "${DRAFTRETRIEVER_CXX_LIB}"
)

add_library(draftretriever_lib SHARED IMPORTED)
set_target_properties(draftretriever_lib PROPERTIES 
    IMPORTED_IMPLIB "${DRAFTRETRIEVER_LIB}"
    IMPORTED_LOCATION "${DRAFTRETRIEVER_DIR}/draftretriever.dll"
)

# Link everything together
target_link_libraries(${TARGET_REST} PRIVATE 
    common 
    llama 
    rest_index
    draftretriever_cxx
    draftretriever_lib
)
target_compile_features(${TARGET_REST} PRIVATE cxx_std_17)

# Add the load-datastore executable for Arrow file handling
set(TARGET_LOAD_DATASTORE load-datastore)
add_executable(${TARGET_LOAD_DATASTORE} load_datastore.cpp)
install(TARGETS ${TARGET_LOAD_DATASTORE} RUNTIME)

# Find Arrow package (this requires Arrow to be installed on your system)
find_package(Arrow REQUIRED)

# Link Arrow libraries
target_link_libraries(${TARGET_LOAD_DATASTORE} PRIVATE 
    arrow 
    arrow_dataset
    parquet
    ${CMAKE_THREAD_LIBS_INIT}
)
target_compile_features(${TARGET_LOAD_DATASTORE} PRIVATE cxx_std_17)
