/opt/cvbml/softwares/ansys_inc/v251/fluent/bin/fluent-cleanup.pl cvbml02.kaist.ac.kr 35627 CLEANUP_EXITING

LOCALHOST=`hostname -s`
if [[ cvbml02.kaist.ac.kr == "$LOCALHOST"* ]]; then kill -9 1541855; else ssh cvbml02.kaist.ac.kr kill -9 1541855; fi
if [[ cvbml02.kaist.ac.kr == "$LOCALHOST"* ]]; then kill -9 1541430; else ssh cvbml02.kaist.ac.kr kill -9 1541430; fi
if [[ cvbml02.kaist.ac.kr == "$LOCALHOST"* ]]; then kill -9 1541226; else ssh cvbml02.kaist.ac.kr kill -9 1541226; fi

rm -f /home/jiwoo/repo/urp-2025/cleanup-fluent-cvbml02.kaist.ac.kr-1541430.sh
