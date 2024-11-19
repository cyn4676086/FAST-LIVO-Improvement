# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "fast_livo: 2 messages, 0 services")

set(MSG_I_FLAGS "-Ifast_livo:/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(fast_livo_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_custom_target(_fast_livo_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "fast_livo" "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" ""
)

get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_custom_target(_fast_livo_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "fast_livo" "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fast_livo
)
_generate_msg_cpp(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fast_livo
)

### Generating Services

### Generating Module File
_generate_module_cpp(fast_livo
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fast_livo
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(fast_livo_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(fast_livo_generate_messages fast_livo_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_cpp _fast_livo_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_cpp _fast_livo_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fast_livo_gencpp)
add_dependencies(fast_livo_gencpp fast_livo_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fast_livo_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fast_livo
)
_generate_msg_eus(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fast_livo
)

### Generating Services

### Generating Module File
_generate_module_eus(fast_livo
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fast_livo
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(fast_livo_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(fast_livo_generate_messages fast_livo_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_eus _fast_livo_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_eus _fast_livo_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fast_livo_geneus)
add_dependencies(fast_livo_geneus fast_livo_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fast_livo_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fast_livo
)
_generate_msg_lisp(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fast_livo
)

### Generating Services

### Generating Module File
_generate_module_lisp(fast_livo
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fast_livo
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(fast_livo_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(fast_livo_generate_messages fast_livo_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_lisp _fast_livo_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_lisp _fast_livo_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fast_livo_genlisp)
add_dependencies(fast_livo_genlisp fast_livo_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fast_livo_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fast_livo
)
_generate_msg_nodejs(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fast_livo
)

### Generating Services

### Generating Module File
_generate_module_nodejs(fast_livo
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fast_livo
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(fast_livo_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(fast_livo_generate_messages fast_livo_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_nodejs _fast_livo_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_nodejs _fast_livo_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fast_livo_gennodejs)
add_dependencies(fast_livo_gennodejs fast_livo_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fast_livo_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo
)
_generate_msg_py(fast_livo
  "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo
)

### Generating Services

### Generating Module File
_generate_module_py(fast_livo
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(fast_livo_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(fast_livo_generate_messages fast_livo_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/Pose6D.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_py _fast_livo_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/elon/ws_FAST-LIVO/src/FAST-LIVO/msg/States.msg" NAME_WE)
add_dependencies(fast_livo_generate_messages_py _fast_livo_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fast_livo_genpy)
add_dependencies(fast_livo_genpy fast_livo_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fast_livo_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fast_livo)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fast_livo
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(fast_livo_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fast_livo)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fast_livo
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(fast_livo_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fast_livo)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fast_livo
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(fast_livo_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fast_livo)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fast_livo
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(fast_livo_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fast_livo
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(fast_livo_generate_messages_py geometry_msgs_generate_messages_py)
endif()
