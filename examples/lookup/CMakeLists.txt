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
add_executable(llama-lookup-rest lookup-rest.cpp)
target_link_libraries(llama-lookup-rest PRIVATE common llama rest_index)
