add_library(GaussQuad INTERFACE
  OrthogonalPolynomial.h
  GaussQuad.h
)

target_include_directories(GaussQuad INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/GaussQuad>
  )
