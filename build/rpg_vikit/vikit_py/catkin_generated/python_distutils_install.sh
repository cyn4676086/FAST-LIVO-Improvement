#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/elon/ws_FAST-LIVO/src/rpg_vikit/vikit_py"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/elon/ws_FAST-LIVO/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/elon/ws_FAST-LIVO/install/lib/python3/dist-packages:/home/elon/ws_FAST-LIVO/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/elon/ws_FAST-LIVO/build" \
    "/usr/bin/python3" \
    "/home/elon/ws_FAST-LIVO/src/rpg_vikit/vikit_py/setup.py" \
     \
    build --build-base "/home/elon/ws_FAST-LIVO/build/rpg_vikit/vikit_py" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/elon/ws_FAST-LIVO/install" --install-scripts="/home/elon/ws_FAST-LIVO/install/bin"
