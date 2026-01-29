cmake_minimum_required(VERSION 3.29)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_FLAGS_DEBUG "-O3")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
project(spontaneous)

find_package(Boost)
#find_package(Matplot++ CONFIG REQUIRED)
find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
find_package(FFTW3l CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
#find_package(SDL2_mixer CONFIG REQUIRED)
#find_package(SDL2_image CONFIG REQUIRED)
#find_package(SDL2_ttf CONFIG REQUIRED)
#find_package(Vulkan REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
#find_package(Matplot++ CONFIG REQUIRED)
find_package(plplot CONFIG REQUIRED)
    
add_executable(spontaneous spdc.cpp)

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/resources
        ${PROJECT_BINARY_DIR}/resources
        COMMENT "Copying resources into binary directory")

target_link_libraries(spontaneous PRIVATE
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    $<IF:$<TARGET_EXISTS:SDL2::SDL2>,SDL2::SDL2,SDL2::SDL2-static>
    #$<IF:$<TARGET_EXISTS:SDL2_image::SDL2_image>,SDL2_image::SDL2_image,SDL2_image::SDL2_image-static>
    #$<IF:$<TARGET_EXISTS:SDL2_mixer::SDL2_mixer>,SDL2_mixer::SDL2_mixer,SDL2_mixer::SDL2_mixer-static>
    #$<IF:$<TARGET_EXISTS:SDL2_ttf::SDL2_ttf>,SDL2_ttf::SDL2_ttf,SDL2_ttf::SDL2_ttf-static>
)
#target_link_libraries(spontaneous PRIVATE Vulkan::Vulkan)
target_link_libraries(spontaneous PRIVATE Boost::boost)
#target_link_libraries(spontaneous PRIVATE Matplot++::cimg  Matplot++::matplot)
target_link_libraries(spontaneous PRIVATE FFTW3::fftw3) #not sure if you have to write static or non static
target_link_libraries(spontaneous PRIVATE FFTW3::fftw3f)
target_link_libraries(spontaneous PRIVATE Eigen3::Eigen)
target_link_libraries(spontaneous PRIVATE FFTW3::fftw3l)
#target_link_libraries(spontaneous PRIVATE Matplot++::cimg  Matplot++::matplot)
#target_link_libraries(GlassesGame PRIVATE Vulkan::dxc)
# C++ binding
target_link_libraries(spontaneous PRIVATE PLPLOT::plplotcxx)
# wxWidgets GUI binding (optional, feature "wxwidgets")
# target_link_libraries(spontaneous PRIVATE PLPLOT::plplotwxwidgets)
# CSIRO Cubic Spline Approximation Library
target_link_libraries(spontaneous PRIVATE PLPLOT::csirocsa)
# QSAS Time Format Conversion Library
target_link_libraries(spontaneous PRIVATE PLPLOT::qsastime)


add_dependencies(spontaneous copy_resources)

