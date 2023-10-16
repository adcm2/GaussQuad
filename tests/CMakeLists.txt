include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_executable(Tests
               Tests.cpp)
target_link_libraries(Tests PRIVATE GaussQuad gtest_main Interpolation)
include(GoogleTest)
gtest_discover_tests(Tests)
