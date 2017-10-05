#! /bin/bash

../linux_sys_int/system_integration.x86_64 &
(sleep 5; rqt)&
(sleep 6; rqt_console)&

roslaunch launch/styx.launch 
