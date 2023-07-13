add_library (GaussQuad INTERFACE GaussQuad.h)

target_link_libraries(GaussQuad INTERFACE Eigen3::Eigen)

target_include_directories (GaussQuad INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
)
