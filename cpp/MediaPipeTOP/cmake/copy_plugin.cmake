# Post-build copy of a plugin file into <repo>/plugins/.
#
# If TouchDesigner has the old DLL loaded (Windows locks loaded DLLs against writes but NOT
# against renames), move it aside as <name>.old-<time> and copy the new one in. The new
# build takes effect the next time TD loads the plugin (restart TD for Custom OPs; toggle
# Unload Plugin for a CPlusPlus TOP). Stale .old-* files are cleaned up once TD lets go.
get_filename_component(dst_dir "${DST}" DIRECTORY)
get_filename_component(dst_name "${DST}" NAME)
file(MAKE_DIRECTORY "${dst_dir}")

file(GLOB stale "${dst_dir}/${dst_name}.old-*")
foreach(f ${stale})
	file(REMOVE "${f}")	# silently fails while still loaded
endforeach()

file(COPY_FILE "${SRC}" "${DST}" RESULT result ONLY_IF_DIFFERENT)
if(result)
	string(TIMESTAMP stamp "%H%M%S")
	file(RENAME "${DST}" "${DST}.old-${stamp}" RESULT rename_result)
	if(rename_result)
		message(WARNING "Could not replace ${DST}: ${result} / ${rename_result}")
	else()
		file(COPY_FILE "${SRC}" "${DST}")
		message(STATUS "${dst_name} was in use; moved the old one aside and copied the new build. Restart TD (Custom OP) or toggle Unload Plugin to pick it up.")
	endif()
else()
	message(STATUS "Copied ${dst_name} to ${dst_dir}")
endif()
