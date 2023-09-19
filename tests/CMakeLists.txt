include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


include(FetchContent)
FetchContent_Declare(
  Interp
  GIT_REPOSITORY https://github.com/da380/Interp.git
  GIT_TAG main
)
FetchContent_MakeAvailable(Interp)




add_executable(Tests
               Tests.cpp)
target_link_libraries(Tests PRIVATE GaussQuad gtest_main Interp)
include(GoogleTest)
gtest_discover_tests(Tests)
