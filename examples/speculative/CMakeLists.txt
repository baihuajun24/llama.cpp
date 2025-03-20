set(TARGET llama-speculative)
add_executable(${TARGET} speculative.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Add new target
set(TARGET_NGRAM ngram-speculative)
add_executable(${TARGET_NGRAM} ngram-speculative.cpp)
install(TARGETS ${TARGET_NGRAM} RUNTIME)
target_link_libraries(${TARGET_NGRAM} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET_NGRAM} PRIVATE cxx_std_17)