cmake_minimum_required(VERSION 3.10)

project(rknn_yolo26_pose_demo)

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
    rknpu3/yolo26.cc
)

add_library(yolo_utils STATIC ${SHARED_SOURCES})
target_include_directories(yolo_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LIBRKNNAPI_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${PROJECT_NAME}
    main.cc
)

add_executable(rknn_yolo26_pose_dataset_eval
    dataset_eval.cc
)

target_link_libraries(${PROJECT_NAME}
    yolo_utils
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNAPI}
    dl
)

target_link_libraries(rknn_yolo26_pose_dataset_eval
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
    target_link_libraries(rknn_yolo26_pose_dataset_eval
        log
    )
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
    target_link_libraries(rknn_yolo26_pose_dataset_eval Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)

target_include_directories(rknn_yolo26_pose_dataset_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNAPI_INCLUDES}
)


install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(TARGETS rknn_yolo26_pose_dataset_eval DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/little_boy.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
file(GLOB WEIGHT_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.weight")
install(FILES ${RKNN_FILES} DESTINATION model)
install(FILES ${WEIGHT_FILES} DESTINATION model)
