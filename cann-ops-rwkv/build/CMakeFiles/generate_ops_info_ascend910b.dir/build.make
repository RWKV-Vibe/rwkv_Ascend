# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Main/cann-ops-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Main/cann-ops-master/build

# Utility rule file for generate_ops_info_ascend910b.

# Include the progress variables for this target.
include CMakeFiles/generate_ops_info_ascend910b.dir/progress.make

CMakeFiles/generate_ops_info_ascend910b: autogen/aic-ascend910b-ops-info.json


autogen/aic-ascend910b-ops-info.json:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/root/Main/cann-ops-master/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating autogen/aic-ascend910b-ops-info.json"
	/usr/local/python3.9.2/bin/python3.9 /usr/local/Ascend/ascend-toolkit/latest/tools/op_project_templates/ascendc/customize/cmake/util/parse_ini_to_json.py /root/Main/cann-ops-master/build/autogen/aic-ascend910b-ops-info.ini /root/Main/cann-ops-master/build/autogen/inner/aic-ascend910b-ops-info.ini /root/Main/cann-ops-master/build/autogen/exc/aic-ascend910b-ops-info.ini /root/Main/cann-ops-master/build/autogen/aic-ascend910b-ops-info.json
	mkdir -p /root/Main/cann-ops-master/build/custom/op_impl/ai_core/tbe/config/ascend910b
	cp -f /root/Main/cann-ops-master/build/autogen/aic-ascend910b-ops-info.json /root/Main/cann-ops-master/build/custom/op_impl/ai_core/tbe/config/ascend910b

generate_ops_info_ascend910b: CMakeFiles/generate_ops_info_ascend910b
generate_ops_info_ascend910b: autogen/aic-ascend910b-ops-info.json
generate_ops_info_ascend910b: CMakeFiles/generate_ops_info_ascend910b.dir/build.make

.PHONY : generate_ops_info_ascend910b

# Rule to build all files generated by this target.
CMakeFiles/generate_ops_info_ascend910b.dir/build: generate_ops_info_ascend910b

.PHONY : CMakeFiles/generate_ops_info_ascend910b.dir/build

CMakeFiles/generate_ops_info_ascend910b.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/generate_ops_info_ascend910b.dir/cmake_clean.cmake
.PHONY : CMakeFiles/generate_ops_info_ascend910b.dir/clean

CMakeFiles/generate_ops_info_ascend910b.dir/depend:
	cd /root/Main/cann-ops-master/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Main/cann-ops-master /root/Main/cann-ops-master /root/Main/cann-ops-master/build /root/Main/cann-ops-master/build /root/Main/cann-ops-master/build/CMakeFiles/generate_ops_info_ascend910b.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/generate_ops_info_ascend910b.dir/depend

