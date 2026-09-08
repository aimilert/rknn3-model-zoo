cmake_minimum_required(VERSION 3.10)

project(rknn_yolov5_demo)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set(SHARED_SOURCES
    postprocess.cc
    yolov5.cc
)

add_library(yolo_utils STATIC ${SHARED_SOURCES})
target_include_directories(yolo_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNAPI_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)

target_link_libraries(${PROJECT_NAME}
    yolo_utils
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNAPI}
    dl
)

add_executable(dataset_eval
    dataset_eval.cc
)

target_include_directories(dataset_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)
target_link_libraries(dataset_eval
    yolo_utils
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNAPI}
    dl
)

# Multi-board / multi-channel YOLOv5 performance benchmark (4 boards x 8 cores)
add_executable(rknn_multicam_demo
    multi_camera_demo.cc
    channel_worker.cc
)

target_include_directories(rknn_multicam_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)
target_link_libraries(rknn_multicam_demo
    yolo_utils
    imageutils
    fileutils
    imagedrawing
    ${LIBRKNNAPI}
    dl
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        log
    )
    target_link_libraries(dataset_eval
        log
    )
endif()

message(STATUS "!!!!!!!!!!!CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
    target_link_libraries(dataset_eval Threads::Threads)
    target_link_libraries(rknn_multicam_demo Threads::Threads)
endif()

install(TARGETS ${PROJECT_NAME} dataset_eval rknn_multicam_demo DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
file(GLOB WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight")
install(FILES ${RKNN_FILES} DESTINATION model)
install(FILES ${WEIGHT_FILES} DESTINATION model)
# Install postprocess plugin to lib directory
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/libpostprocess_rk182x/prebuilt/libpostprocess_yolov5_rk182x.so DESTINATION lib)